#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :		run_omiclm_train.py
@Time    :   	2024/05/23 16:47:00
@Author  :   	Yong Bai 
@Contact :   	baiyong at genomics.cn
@License :   	(C)Copyright 2023-2024, Yong Bai

                Licensed under the Apache License, Version 2.0 (the "License");
                you may not use this file except in compliance with the License.
                You may obtain a copy of the License at

                    http://www.apache.org/licenses/LICENSE-2.0

                Unless required by applicable law or agreed to in writing, software
                distributed under the License is distributed on an "AS IS" BASIS,
                WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
                See the License for the specific language governing permissions and
                limitations under the License.

@Desc    :   	None

"""

import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

import sys
from typing import Any, Dict, List, Mapping, NewType
import logging
from pathlib import Path
import json
import math
import copy
import gc

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
import transformers
import datasets
import evaluate
from tqdm import tqdm
from dataclasses import dataclass
import scanpy as sc
 
from transformers import (
    HfArgumentParser,
    EvalPrediction,
    set_seed
)

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from transformers.utils import check_min_version
from transformers.trainer_utils import get_last_checkpoint

sys.path.append(
    str(Path(__file__).resolve().parents[1]) #
)
from omic_config import (
    OmicRawDataConfig,
    OmicPretrainedModelAndTokenizationConfig,
    OmicFormerTrainingArguments,
    OmicFormerTrainer,
)

from omic_models import (
    OmicFormerConfig, 
    OmicFormerPreTrainedModel,

    PretrainedModelName,
    PRETRAINED_TOKENIZER_NAME_CLS_MAP,
)

from omic_dataset_collator import OmicDataset, OmicDataCollator

check_min_version("4.39.3")
logger = logging.getLogger(__name__)

def main():
    parser = HfArgumentParser((
        OmicRawDataConfig, 
        OmicPretrainedModelAndTokenizationConfig, 
        OmicFormerTrainingArguments))
    raw_data_config, pretrained_model_tokenizer_config, training_config = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    set_seed(training_config.seed)

    if training_config.should_log:
        # The default of training_args.log_level is passive, 
        # so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_config.get_process_log_level() # log_level = 20
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_config.local_rank}, device: {training_config.device}, n_gpu: {training_config.n_gpu}, "
        + f"distributed training: {training_config.parallel_mode.value == 'distributed'}, 16-bits training: {training_config.fp16}"
    )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_config.output_dir) and training_config.do_train and not training_config.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_config.output_dir)
        if last_checkpoint is None and len(os.listdir(training_config.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_config.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_config.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    #################################################################
    # load TOKENIZED SEQ dataset and SCRNA embedding data
    #################################################################
    raw_scrna_count_file_name = raw_data_config.raw_scrna_file_name
    adata_file_base_name = os.path.basename(raw_scrna_count_file_name).split('.')[0]

    if pretrained_model_tokenizer_config.seq_model_name == PretrainedModelName.HYENADNA:
        tokenizer_cls = PRETRAINED_TOKENIZER_NAME_CLS_MAP[pretrained_model_tokenizer_config.seq_model_name]
        tokenizer = tokenizer_cls(
            characters=['A', 'C', 'G', 'T', 'N'],  # add DNA characters, 7, 8, 9, 10, 11
            model_max_length=pretrained_model_tokenizer_config.seq_model_input_max_len
        )
    else:
        tokenizer = None
    
    with training_config.main_process_first(desc="loading seq data and model"):

        seq_tokenized_dataset_dir = pretrained_model_tokenizer_config.tokenized_seq_dataset_dir
        logger.info(f">>> loading TOKENIZED SEQ data and split from {seq_tokenized_dataset_dir}")
        seq_hf_ds = load_from_disk(seq_tokenized_dataset_dir)

        logger.info(f"seq tokenized datasets: \n{seq_hf_ds}")
        # >>> SEQ tokenized data:
        # DatasetDict({
        #     train: Dataset({
        #         features: ['sample', 'pos', 'seq', 'peak_value', 'input_ids', 'attention_mask', 'record_id'],
        #         num_rows: 124167712
        #     })
        #     test: Dataset({
        #         features: ['sample', 'pos', 'seq', 'peak_value', 'input_ids', 'attention_mask', 'record_id'],
        #         num_rows: 4208429
        #     })
        # })

        logger.info(f">>> spliting TRAIN DATASET into train (99.95%) and valiadtion (0.05%) datasets.")
        train_val_dataset = seq_hf_ds['train'].train_test_split(test_size=0.0005, shuffle=True, seed=training_config.data_seed)
        
        _trn_dataset = train_val_dataset['train']
        if pretrained_model_tokenizer_config.use_streaming:
            trn_dataset = _trn_dataset.to_iterable_dataset(num_shards=len(_trn_dataset))
        else:
            trn_dataset = copy.deepcopy(_trn_dataset)
        del _trn_dataset
        gc.collect()
        logger.info(f"train dataset: \n{trn_dataset}")

        # Dataset({
        #     features: ['sample', 'pos', 'seq', 'peak_value', 'input_ids', 'attention_mask', 'record_id'],
        #     num_rows: 111750940
        # })

        _val_dataset = train_val_dataset['test']
        if pretrained_model_tokenizer_config.use_streaming:
            val_dataset = _val_dataset.to_iterable_dataset(num_shards=len(_val_dataset))
        else:
            val_dataset = copy.deepcopy(_val_dataset)
        del _val_dataset
        gc.collect()
        logger.info(f"validation dataset: \n{val_dataset}")
        # Dataset({
        #     features: ['sample', 'pos', 'seq', 'peak_value', 'input_ids', 'attention_mask', 'record_id'],
        #     num_rows: 12416772
        # })

        # # this is only for precidtion
        # tst_dataset = seq_hf_ds['test']
        # logger.info(f"test dataset: \n{tst_dataset}")
        # # Dataset({
        # #     features: ['sample', 'pos', 'seq', 'peak_value', 'input_ids', 'attention_mask', 'record_id'],
        # #     num_rows: 4208429
        # # })

    if training_config.do_train:
        if training_config.max_train_samples is not None:
            if not pretrained_model_tokenizer_config.use_streaming:
                trn_dataset =  trn_dataset.select(range(training_config.max_train_samples))
            else:
                trn_dataset = trn_dataset.take(training_config.max_train_samples)

    if training_config.do_eval:
        logger.info(f">>> loading EVAL DATASET")
        if training_config.max_eval_samples is not None:
            if not pretrained_model_tokenizer_config.use_streaming:
                val_dataset = val_dataset.select(range(training_config.max_eval_samples))
            else:
                val_dataset = val_dataset.take(training_config.max_eval_samples)
        
        def preprocess_logits_for_metrics(logits, labels):

            if isinstance(logits, tuple):
                logits = logits[0]
            return logits
        
        metric_mse = evaluate.load("metrics/mse")
        metric_r = evaluate.load("metrics/pearsonr")
        # metric = evaluate.MetricWrapper(metric, preprocess_logits_for_metrics)
        # metric_name = "mse"
        # metric_args = {"metric_name": metric_name, "metric": metric}
        def compute_metrics_fn(eval_pred: EvalPrediction):
            logits, ture_peak = eval_pred

            return {**metric_mse.compute(predictions=logits, references=ture_peak),
                    **metric_r.compute(predictions=logits, references=ture_peak, return_pvalue=True)}
    

    scrna_embedding_dataset_dir = os.path.join(pretrained_model_tokenizer_config.emb_scrna_dataset_dir, adata_file_base_name)
    
    # logger.info(f">>> SCRNA EMBEDDING dataset:\n{scrna_embedding_dataset_trn}")
    ac_data_file_name = os.path.join(scrna_embedding_dataset_dir, f"{adata_file_base_name}_embedding.h5ad")
    logger.info(f">>> loading SCRNA EMBEDDING data (h5ad) from {ac_data_file_name}")
    sc_adata = sc.read_h5ad(ac_data_file_name)
    
    trn_pt_ds: torch.utils.data.Dataset = OmicDataset(trn_dataset, sc_adata)
    val_pt_ds: torch.utils.data.Dataset = OmicDataset(val_dataset, sc_adata)

    seq_input_pooling_size = 512
    scrna_input_pooling_size = 200
    
    omic_data_collator = OmicDataCollator(seq_pad_token_id=4, sc_max_cnt=scrna_input_pooling_size, seq_max_len=seq_input_pooling_size)

    omicformer_config = dict(
        seq_model_name=pretrained_model_tokenizer_config.seq_model_name.value,
        seq_pretrained_model_name_or_path=pretrained_model_tokenizer_config.seq_model_path,
        seq_emb_dim=256,
        seq_emb_extraction_kwargs=None,

        scrna_model_name=pretrained_model_tokenizer_config.scrna_model_name.value,
        scrna_pretrained_model_name_or_path=pretrained_model_tokenizer_config.scrna_model_path,
        scrna_emb_dim=512, 
        scrna_emb_extraction_kwargs=None,

        seq_emb_input=False,
        scrna_emb_input=True,

        seq_input_pooling_size=seq_input_pooling_size,     # pooling seq length after seq embedding by pretrained model 
        scrna_input_pooling_size=scrna_input_pooling_size,   # number of cells after pooling after scrna embedding by pretrained model
        
        seq_input_pooling_mode='adaptive',   # fisrt, last, mean, weight, adaptive, max
        scrna_input_pooling_mode='adaptive', # fisrt, last, mean, weight, adaptive, max

        pre_layer_type='gated_mlp',  # 'gated_mlp', 'mlp

        ffn_type=pretrained_model_tokenizer_config.attn_ffn_type, # 'moe', 'mlp
        hidden_dim=512,
        intermediate_hidden_dim=768,
        
        n_layers_encoder=4,
        
        n_heads=8, 
        num_experts=6,
        moe_topk=2,
        dropout=0.1,

        initializer_range=0.02,

        fusion_type='cross_attn',   # 'cross_attn', 'clip_style'(not implemented yet) 
        
        n_layers_fusion=8,

        n_residuals_per_layer=2,     # Change to 2 if we have MLP, otherwise 1. Not used for now

        out_pooling_size=4,
        out_pooling_mode='adaptive',   # fisrt, last, mean, weight, adaptive, max
        n_outputs= 1,
    )
    omic_model = OmicFormerPreTrainedModel(OmicFormerConfig(**omicformer_config))
    omic_model.config.use_cache = False
    logger.info(f">>> OmicFormerConfig: \n{json.dumps(omicformer_config, indent=2, sort_keys=True)}")
    logger.info(f">>> OmicFormerPreTrainedModel: \n{omic_model}")

    logger.info(f"^^^^^^^^tf32 is set: {torch.backends.cuda.matmul.allow_tf32}")
    logger.info(f"^^^^^^^^fp16 = {training_config.fp16}")
    logger.info(f"^^^^^^^^Learning rate: {training_config.learning_rate}")
    logger.info(f"^^^^^^^^LR scheduler type : {training_config.lr_scheduler_type}")
    logger.info(f"^^^^^^^^use streaming : {pretrained_model_tokenizer_config.use_streaming}")

    trainer = OmicFormerTrainer(
        model=omic_model,
        args=training_config,
        train_dataset=trn_pt_ds if training_config.do_train else None,
        eval_dataset=val_pt_ds if training_config.do_eval else None,
        tokenizer=tokenizer,
        data_collator=omic_data_collator,
        compute_metrics=compute_metrics_fn if training_config.do_eval else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if training_config.do_eval else None,
    )

    # trainer = OmicFormerTrainer(
    #     model=omic_model,
    #     args=training_config,
    #     train_dataset=trn_pt_ds if training_config.do_train else None, 
    #     tokenizer=tokenizer,
    #     data_collator=omic_data_collator,
    # )

    ####################################################################
    # Training and evaluation
    #
    ####################################################################
    logger.info(">>>>>>>>>>>>>>>>Start training and evaluatoin......")
    if training_config.do_train:
        checkpoint = None
        if training_config.resume_from_checkpoint is not None:
            checkpoint = training_config.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        trainer.save_model(training_config.output_dir)  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        max_train_samples = (
            training_config.max_train_samples if training_config.max_train_samples is not None else len(trn_pt_ds)
        )

        metrics["train_samples"] = min(max_train_samples, len(trn_pt_ds))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_config.do_eval:
        metrics = trainer.evaluate()

        max_eval_samples = training_config.max_eval_samples if training_config.max_eval_samples is not None else len(val_pt_ds)
        
        metrics["eval_samples"] = min(max_eval_samples, len(val_pt_ds))
        
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    with training_config.main_process_first(desc="clean scRNA cache files"):
        seq_hf_ds.cleanup_cache_files()

    logger.info("<<<<<<<<<<<<<<<<Done")


if __name__=='__main__':
    main()


