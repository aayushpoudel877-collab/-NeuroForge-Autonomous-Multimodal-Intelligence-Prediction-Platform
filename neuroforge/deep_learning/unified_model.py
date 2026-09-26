from __future__ import annotations
from torch import nn
from .modal_encoders import TextTransformerEncoder,VisionCNNEncoder,AudioCNNEncoder,TemporalTransformerEncoder
from .cross_modal import CrossModalAttention

class NeuroForgeMultimodalModel(nn.Module):
    def __init__(self,embedding_dim=128):
        super().__init__()
        self.text=TextTransformerEncoder(embedding_dim=embedding_dim)
        self.vision=VisionCNNEncoder(embedding_dim=embedding_dim)
        self.audio=AudioCNNEncoder(embedding_dim=embedding_dim)
        self.temporal=TemporalTransformerEncoder(embedding_dim=embedding_dim)
        self.cross_modal=CrossModalAttention(embedding_dim=embedding_dim)
        self.head=nn.Sequential(
            nn.Linear(embedding_dim,embedding_dim),nn.GELU(),nn.Dropout(.1),
            nn.Linear(embedding_dim,1)
        )
    def forward(self,text_tokens,images,audio,series):
        embeddings=[self.text(text_tokens),self.vision(images),self.audio(audio),self.temporal(series)]
        fused=self.cross_modal(embeddings).mean(dim=1)
        return self.head(fused).squeeze(-1)
