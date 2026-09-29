from __future__ import annotations
import torch
from torch import nn
from .modal_encoders import TextTransformerEncoder,VisionCNNEncoder,AudioCNNEncoder,TemporalTransformerEncoder
from .cross_modal import CrossModalAttention

class NeuroForgeMultimodalModel(nn.Module):
    """Four-modality model with learned embeddings for unavailable modalities."""
    MODALITIES=("text","image","audio","temporal")

    def __init__(self,embedding_dim=128):
        super().__init__()
        self.text=TextTransformerEncoder(embedding_dim=embedding_dim)
        self.vision=VisionCNNEncoder(embedding_dim=embedding_dim)
        self.audio=AudioCNNEncoder(embedding_dim=embedding_dim)
        self.temporal=TemporalTransformerEncoder(embedding_dim=embedding_dim)
        self.missing_embeddings=nn.Parameter(torch.zeros(4,embedding_dim))
        self.cross_modal=CrossModalAttention(embedding_dim=embedding_dim)
        self.head=nn.Sequential(nn.Linear(embedding_dim,embedding_dim),nn.GELU(),nn.Dropout(.1),nn.Linear(embedding_dim,1))

    def forward(self,text_tokens,images,audio,series,modality_mask=None):
        embeddings=[self.text(text_tokens),self.vision(images),self.audio(audio),self.temporal(series)]
        stacked=torch.stack(embeddings,dim=1)
        if modality_mask is not None:
            mask=torch.as_tensor(modality_mask,device=stacked.device,dtype=torch.bool)
            if mask.shape!=stacked.shape[:2]:
                raise ValueError("modality_mask must have shape [batch, 4]")
            missing=self.missing_embeddings.unsqueeze(0).expand(stacked.shape[0],-1,-1)
            stacked=torch.where(mask.unsqueeze(-1),stacked,missing)
        fused=self.cross_modal(stacked).mean(dim=1)
        return self.head(fused).squeeze(-1)
