from __future__ import annotations
import torch
from torch import nn

class CrossModalAttention(nn.Module):
    def __init__(self,embedding_dim=128,heads=4):
        super().__init__()
        self.attention=nn.MultiheadAttention(embedding_dim,heads,batch_first=True)
        self.norm=nn.LayerNorm(embedding_dim)
        self.gate=nn.Sequential(nn.Linear(embedding_dim,embedding_dim),nn.Sigmoid())

    def forward(self,modal_embeddings):
        tokens=modal_embeddings if isinstance(modal_embeddings,torch.Tensor) else torch.stack(modal_embeddings,dim=1)
        if tokens.ndim!=3: raise ValueError("modal embeddings must have shape [batch, modalities, embedding]")
        attended,_=self.attention(tokens,tokens,tokens)
        gated=attended*self.gate(attended)
        return self.norm(tokens+gated)
