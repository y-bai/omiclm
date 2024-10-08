#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :		_loss.py
@Time    :   	2024/06/07 09:48:45
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

import torch
import torch.nn as nn
import torch.nn.functional as F


class MSLELoss(nn.Module):
    """
    Mean Squared Logarithmic Error

    """
    def __init__(self) -> None:
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        msle = self.mse(torch.log1p(input), torch.log1p(target))

        return msle
    

class MSEMSLELoss(nn.Module):
    """
    Mean Squared Logarithmic Error

    """
    def __init__(self, alpha=2.0, beta=10.0, auxiliary_loss='mse') -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta

        self.msle_loss = MSLELoss()

        if auxiliary_loss == 'mse':
            self.au_loss = nn.MSELoss()
        else:
            self.au_loss = nn.L1Loss()
        
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        loss = self.alpha * self.au_loss(input, target) + self.beta * self.msle_loss(input, target)
        # loss = self.alpha * au_loss
        return loss
    
class CLIPLoss(nn.Module):
    """
    CLIP loss

    Adapted from:

    """

    def __init__(
        self,
        temprature: float = 0.07, # default value from CLIP paper, or 2.6592
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

        self.temprature = nn.Parameter(
            torch.ones([]) * temprature, requires_grad=False
        )
    def forward(self, seq_embedding: torch.Tensor, scrna_embedding: torch.Tensor) -> torch.Tensor:
        # seq_embedding: (batch_size, seq_lenth, embeding_dim)
        # scrna_embedding: (batch_size, sc_cnt, embeding_dim)

        if seq_embedding.dim() == 3:
            seq_embedding, _ = seq_embedding.max(dim=1)       # (n_seq [or batch size], embeding_dim)
        if scrna_embedding.dim() == 3:
            scrna_embedding, _ = scrna_embedding.max(dim=1)   # (n_sc [or batch_size], embeding_dim)

        loss = self._clip_loss(seq_embedding, scrna_embedding)

        return loss
    
    def _clip_loss(
        self,
        seq_embedding: torch.Tensor,
        scrna_embedding: torch.Tensor,
    ) -> torch.Tensor:
        # similarity: (n_sc, n_seq)
        similarity_logit = (seq_embedding @ scrna_embedding.t()) * torch.exp(self.temprature)   # (n_seq, n_sc)
        seq_loss = self._contrastive_loss_v1(similarity_logit, dim=0)
        scrna_loss = self._contrastive_loss_v1(similarity_logit, dim=1)
        # seq_loss = self._contrastive_loss_v2(similarity_logit)
        # scrna_loss = self._contrastive_loss_v2(similarity_logit.t())
        return (seq_loss + scrna_loss) / 2

    # Adapted from https://sachinruk.github.io/blog/2021-03-07-clip.html
    def _contrastive_loss_v1(self, logits, dim):
        neg_ce = torch.diag(F.log_softmax(logits, dim=dim))
        return -neg_ce.mean()
    
    def _contrastive_loss_v2(self, logits):
        return F.cross_entropy(logits, torch.arange(logits.shape[0], device=logits.device))


