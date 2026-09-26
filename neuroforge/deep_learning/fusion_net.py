from __future__ import annotations
import torch
from torch import nn
from .encoders import MLPEncoder

class MultimodalFusionNet(nn.Module):
    """Trainable late-fusion network for four modality vectors."""
    def __init__(self,input_dims:dict[str,int],embedding_dim:int=64):
        super().__init__()
        self.modalities=list(input_dims)
        self.encoders=nn.ModuleDict({k:MLPEncoder(v,embedding_dim=embedding_dim) for k,v in input_dims.items()})
        self.gate=nn.Sequential(nn.Linear(embedding_dim*len(self.modalities),len(self.modalities)),nn.Softmax(dim=-1))
        self.head=nn.Sequential(nn.Linear(embedding_dim,128),nn.GELU(),nn.Dropout(.15),nn.Linear(128,1))
    def forward(self,features:dict[str,torch.Tensor]):
        embeddings=[self.encoders[k](features[k]) for k in self.modalities]
        stacked=torch.stack(embeddings,dim=1)
        weights=self.gate(torch.cat(embeddings,dim=-1))
        fused=(stacked*weights.unsqueeze(-1)).sum(dim=1)
        return self.head(fused).squeeze(-1),weights
