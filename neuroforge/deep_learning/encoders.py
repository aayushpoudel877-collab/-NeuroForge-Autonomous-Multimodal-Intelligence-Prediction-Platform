from __future__ import annotations
import torch
from torch import nn

class MLPEncoder(nn.Module):
    """Small configurable encoder used for modality feature vectors."""
    def __init__(self,input_dim:int,hidden_dim:int=128,embedding_dim:int=64,dropout:float=0.15):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(input_dim,hidden_dim),nn.LayerNorm(hidden_dim),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden_dim,embedding_dim),nn.GELU())
    def forward(self,x:torch.Tensor)->torch.Tensor:
        return self.net(x)
