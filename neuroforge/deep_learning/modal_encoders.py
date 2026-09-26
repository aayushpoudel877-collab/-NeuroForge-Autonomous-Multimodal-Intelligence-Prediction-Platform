from __future__ import annotations
import torch
from torch import nn

class TextTransformerEncoder(nn.Module):
    def __init__(self,vocab_size=4096,embedding_dim=128,heads=4,layers=2,max_length=128):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,embedding_dim)
        layer=nn.TransformerEncoderLayer(d_model=embedding_dim,nhead=heads,batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(layer,num_layers=layers)
        self.position=nn.Parameter(torch.zeros(1,max_length,embedding_dim))
        self.max_length=max_length
    def forward(self,tokens):
        x=self.embedding(tokens[:,:self.max_length])+self.position[:,:tokens.shape[1]]
        return self.encoder(x).mean(dim=1)

class VisionCNNEncoder(nn.Module):
    def __init__(self,in_channels=3,embedding_dim=128):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(in_channels,32,3,2,1),nn.GELU(),
            nn.Conv2d(32,64,3,2,1),nn.GELU(),
            nn.Conv2d(64,128,3,2,1),nn.GELU(),
            nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(128,embedding_dim),nn.LayerNorm(embedding_dim)
        )
    def forward(self,images):
        return self.net(images)

class AudioCNNEncoder(nn.Module):
    def __init__(self,in_channels=1,embedding_dim=128):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv1d(in_channels,32,7,2,3),nn.GELU(),
            nn.Conv1d(32,64,7,2,3),nn.GELU(),
            nn.Conv1d(64,128,5,2,2),nn.GELU(),
            nn.AdaptiveAvgPool1d(1),nn.Flatten(),nn.Linear(128,embedding_dim),nn.LayerNorm(embedding_dim)
        )
    def forward(self,waveforms):
        return self.net(waveforms)

class TemporalTransformerEncoder(nn.Module):
    def __init__(self,features=1,embedding_dim=128,heads=4,layers=2):
        super().__init__()
        self.projection=nn.Linear(features,embedding_dim)
        layer=nn.TransformerEncoderLayer(d_model=embedding_dim,nhead=heads,batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(layer,num_layers=layers)
    def forward(self,series):
        return self.encoder(self.projection(series)).mean(dim=1)
