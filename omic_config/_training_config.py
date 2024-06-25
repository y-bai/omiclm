#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :		_training_config.py
@Time    :   	2024/05/16 14:10:05
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
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from transformers import TrainingArguments

import sys
sys.path.append("..")

from omic_models import PretrainedModelName

##########################################################
#
# prtrained model name
PRETRAINED_SEQ_MODEL_NAME = PretrainedModelName.HYENADNA
SEQ_MODEL_SIZE_INDICATOR =  450000        # SEQ_MAX_LEN used for selecting pretrained hyenadna model, NOTE: values could be 1024, 32768, 160000, 450000, 1_000_000
SEQ_MIN_LEN = 0                             # for filter seq with length < SEQ_MIN_LEN when tokenizing

PRETRAINED_SCRNA_MODEL_NAME = PretrainedModelName.SCGPT     # scRNA pretrained model
SCRNA_MODEL_SIZE_INDICATOR = 512
##########################################################

RAW_SCRNA_HAS_HVG = True
RAW_SCRNA_CELL_TYPE_VAR_NAME = "celltype_l4"  # This is used for Geneformer when tokenization

USE_STREAMING = False
MSLE_LOSS_ALPHA = 1.0   
MSLE_LOSS_BETA = 5.0    

ATTN_FFN_TYPE = "gated_mlp"  # "moe" or "mlp", "gated_mlp"


DATA_PATH = {
    # raw seq data in .tsv/.csv file
    # NOTE: change to the relative dictionary where raw .tsv/.csv file stored
    "RAW_SEQ_PLAIN_FILE_NAME_FOR_TRAIN": "processed_data/DNA_training_set.csv",
    "RAW_SEQ_PLAIN_FILE_NAME_FOR_TEST": "processed_data/DNA_test_set.csv",

    # scRNA data
    # NOTE: change to the relative dictionary where .h5ad file stored
    "RAW_SCRNA_DATA_FILE_NAME": "processed_data/CD8_expression_5K.h5ad",
    # "RAW_SCRNA_DATA_FILE_NAME": "processed_data/CD8_expression_ensembl.h5ad",

    "RAW_SCRNA_DATA_PREPROCESSED_DIR": "datasets/raw_datasets/scrna_processed",

    # raw seq data after being converted into `datasets.dataset` file
    "RAW_SEQ_DATASET_DIR": "datasets/raw_datasets/seq_dataset",

    # tokenized dataset directory
    "TOKENIZED_SEQ_DATASET_DIR": "datasets/tokenized_datasets/seq_dataset",
    "TOKENIZED_SCRNA_DATASET_DIR": "datasets/tokenized_datasets/scrna_dataset",

    # embedding dataset directory
    "EMBEDDING_SEQ_DATASET_DIR": "datasets/embedding_datasets/seq_dataset",

    # embedding dataset directory
    "EMBEDDING_SCRNA_DATASET_DIR": "datasets/embedding_datasets/scrna_dataset",

    # train output dir
    "TRAINING_OUTPUT_DIR": "outputs",

    "PROJECT_ROOT_PATH": r"path/to/project/root",  
}

PRETRAINED_MODEL_NAME_PATH = {
    PretrainedModelName.HYENADNA: {
        1024: "HyenaDNA/hyenadna-tiny-1k-seqlen",
        32768: "HyenaDNA/hyenadna-small-32k-seqlen",
        160000: "HyenaDNA/hyenadna-medium-160k-seqlen",
        450000: "HyenaDNA/hyenadna-medium-450k-seqlen",
        1_000_000: "HyenaDNA/hyenadna-large-1m-seqlen"
    },
    PretrainedModelName.GENEFORMER: {
        512: "geneformer-12L-30M",  # hidden_size: model_path
    },
    PretrainedModelName.SCGPT: {
        512: "scGPT/scGPT_human",   # emd_size: model_path
    },
}

BATCHE_SIZE = { # ATTN_FFN_TYPE: BATCH_SIZE
    'moe': 16,
    'mlp': 88,
    'gated_mlp': 88,
}


@dataclass
class OmicRawDataConfig:

    raw_scrna_cell_type_var_name: Optional[str] = RAW_SCRNA_CELL_TYPE_VAR_NAME

    # seq raw data plain file (.tsv/.csv)
    raw_seq_plain_file_name_for_train = os.path.join(
        DATA_PATH["PROJECT_ROOT_PATH"], 
        DATA_PATH["RAW_SEQ_PLAIN_FILE_NAME_FOR_TRAIN"])
    raw_seq_plain_file_name_for_test = os.path.join(
        DATA_PATH["PROJECT_ROOT_PATH"], 
        DATA_PATH["RAW_SEQ_PLAIN_FILE_NAME_FOR_TEST"])
    
    # sc rna data (.h5ad, train and test in a single .h5ad file)
    raw_scrna_file_name: Optional[str] = os.path.join(
        DATA_PATH["PROJECT_ROOT_PATH"], 
        DATA_PATH["RAW_SCRNA_DATA_FILE_NAME"])
    
    raw_scrna_has_hvg: Optional[bool] = RAW_SCRNA_HAS_HVG
    
    raw_scrna_preprocessed_dir: Optional[str] = os.path.join(
        DATA_PATH["PROJECT_ROOT_PATH"],
        DATA_PATH["RAW_SCRNA_DATA_PREPROCESSED_DIR"] + f"/{PRETRAINED_SCRNA_MODEL_NAME.value}")

    # raw seq dataset file (datasets.DatasetDict as being converted from the plain file into datasets.DatasetDict with train and test)
    raw_seq_dataset_dir: Optional[str] = os.path.join(
        DATA_PATH["PROJECT_ROOT_PATH"], 
        DATA_PATH["RAW_SEQ_DATASET_DIR"]) 
    


@dataclass
class OmicPretrainedModelAndTokenizationConfig:

    seq_model_name: Optional[PretrainedModelName] = PRETRAINED_SEQ_MODEL_NAME
    seq_model_input_max_len: Optional[int] = SEQ_MODEL_SIZE_INDICATOR

    scrna_model_name: Optional[PretrainedModelName] = PRETRAINED_SCRNA_MODEL_NAME

    use_streaming: bool = USE_STREAMING
    seq_min_len: int = SEQ_MIN_LEN  # for filter seq with length < SEQ_MIN_LEN when tokenizing

    attn_ffn_type: str = ATTN_FFN_TYPE

    # pretrained seq model path
    seq_model_path = os.path.join(
        DATA_PATH["PROJECT_ROOT_PATH"],
        PRETRAINED_MODEL_NAME_PATH[PRETRAINED_SEQ_MODEL_NAME][SEQ_MODEL_SIZE_INDICATOR]
    )
    # pretrained scrna model path
    scrna_model_path: Optional[str] = os.path.join(
        DATA_PATH["PROJECT_ROOT_PATH"],
        PRETRAINED_MODEL_NAME_PATH[PRETRAINED_SCRNA_MODEL_NAME][SCRNA_MODEL_SIZE_INDICATOR]
    )

    # tokenized seq dataset saved path
    tokenized_seq_dataset_dir:Optional[str]= os.path.join(
        DATA_PATH["PROJECT_ROOT_PATH"], 
        DATA_PATH["TOKENIZED_SEQ_DATASET_DIR"] + f"/{PRETRAINED_SEQ_MODEL_NAME.value}" 
    )

    tokenized_scrna_dataset_dir: Optional[str] = os.path.join(
        DATA_PATH["PROJECT_ROOT_PATH"], 
        DATA_PATH["TOKENIZED_SCRNA_DATASET_DIR"] + f"/{PRETRAINED_SCRNA_MODEL_NAME.value}"
    )

    emb_seq_dataset_dir: Optional[str] = os.path.join(
        DATA_PATH["PROJECT_ROOT_PATH"], 
        DATA_PATH["EMBEDDING_SEQ_DATASET_DIR"] + f"/{PRETRAINED_SEQ_MODEL_NAME.value}"
    )

    emb_scrna_dataset_dir: Optional[str] = os.path.join(
        DATA_PATH["PROJECT_ROOT_PATH"], 
        DATA_PATH["EMBEDDING_SCRNA_DATASET_DIR"] + f"/{PRETRAINED_SCRNA_MODEL_NAME.value}"
    )


@dataclass
class OmicFormerTrainingArguments(TrainingArguments):
    output_dir: str = os.path.join(
        DATA_PATH["PROJECT_ROOT_PATH"], 
        DATA_PATH["TRAINING_OUTPUT_DIR"] + f"/{ATTN_FFN_TYPE}")
    overwrite_output_dir: bool = True 

    msle_alpha: float = MSLE_LOSS_ALPHA
    msle_beta: float = MSLE_LOSS_BETA
    auxiliary_loss: str = "mse"
    label_names: List[str] = field(default_factory=lambda: ["peak_value"])

    learning_rate: float = 6e-5   
    lr_scheduler_type: str = "cosine"
    # lr_scheduler_kwargs: Dict[str, Any] = field(default_factory=lambda: {"min_lr": 1e-8})

    warmup_steps: int = 1000 
    max_steps:int = 60000  

    dataloader_num_workers: int = 4      
    per_device_train_batch_size: int = BATCHE_SIZE[ATTN_FFN_TYPE] 
    per_device_eval_batch_size: int = BATCHE_SIZE[ATTN_FFN_TYPE]

    gradient_accumulation_steps: int = 2  

    eval_accumulation_steps: int = 100

    # NOTE: config for evaluation
    evaluation_strategy: str = "steps"
    # evaluation_strategy: str = "no"
    do_eval: bool = True

    #
    # if evaluation_strategy="steps". eval_steps will default to the same 
    # value as logging_steps if not set.
    # eval_steps must be an integer if bigger than 1
    eval_steps: int = 100
    
    # NOTE: logging config 
    # TensorBorad log dir
    logging_dir: str = os.path.join(
        DATA_PATH["PROJECT_ROOT_PATH"], 
        DATA_PATH["TRAINING_OUTPUT_DIR"] + f"/{ATTN_FFN_TYPE}/log")
    logging_steps: int = 100 #
    logging_strategy: str = "steps"
    report_to: str = "tensorboard" 

    # NOTE: save config
    save_steps: int = 2000 
    save_strategy: str = "steps"
    save_total_limit: int = 3

    weight_decay: float = 0.1           
    adam_beta1:float = 0.9              # default for AdamW
    adam_beta2:float = 0.999            # default: 0.999
    adam_epsilon:float = 1e-8

    do_train: bool = True

    max_grad_norm:float = 1.0  # lib defult value

    sharded_ddp: bool = True   # speed up training under multi-GPU
    ddp_timeout: int = 60 * 60 * 2 # 1-hour

    # find_unused_parameters in DistributedDataParallel
    # NOTE
    ddp_find_unused_parameters: bool = False

    resume_from_checkpoint: bool = False 

    seed: int = 42
    data_seed: int = 42

    #
    # # If input does not contained labels, then we need to use this
    # include_inputs_for_metrics: bool = True

    #
    disable_tqdm: bool = True 

    tf32: bool = True
    # fp16: bool = True 
    # gradient_checkpointing: bool = True

    # for debug
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
