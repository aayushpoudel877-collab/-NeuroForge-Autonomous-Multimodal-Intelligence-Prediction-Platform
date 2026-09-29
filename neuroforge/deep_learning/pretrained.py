from __future__ import annotations
import torch
from torch import nn

class HuggingFaceTextEncoder(nn.Module):
    def __init__(self,model_name="distilbert-base-uncased",output_dim=128,freeze_backbone=True):
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise ImportError("install the 'advanced' extra to use HuggingFaceTextEncoder") from exc
        self.backbone=AutoModel.from_pretrained(model_name)
        if freeze_backbone:
            for p in self.backbone.parameters(): p.requires_grad=False
        self.projection=nn.Linear(self.backbone.config.hidden_size,output_dim)

    def forward(self,input_ids,attention_mask=None):
        out=self.backbone(input_ids=input_ids,attention_mask=attention_mask)
        hidden=out.last_hidden_state
        if attention_mask is None: pooled=hidden.mean(dim=1)
        else:
            mask=attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled=(hidden*mask).sum(dim=1)/mask.sum(dim=1).clamp_min(1)
        return self.projection(pooled)

class TorchvisionVisionEncoder(nn.Module):
    def __init__(self,model_name="resnet50",output_dim=128,pretrained=True,freeze_backbone=True):
        super().__init__()
        try:
            import torchvision.models as models
        except ImportError as exc:
            raise ImportError("install the 'advanced' extra to use TorchvisionVisionEncoder") from exc
        self.backbone=models.get_model(model_name,weights="DEFAULT" if pretrained else None)
        if hasattr(self.backbone,"fc"): self.backbone.fc=nn.Identity()
        elif hasattr(self.backbone,"heads") and hasattr(self.backbone.heads,"head"): self.backbone.heads.head=nn.Identity()
        if freeze_backbone:
            for p in self.backbone.parameters(): p.requires_grad=False
        self.projection=nn.LazyLinear(output_dim)

    def forward(self,images):
        features=self.backbone(images)
        if features.ndim>2: features=features.flatten(1)
        return self.projection(features)

class Wav2Vec2AudioEncoder(nn.Module):
    def __init__(self,output_dim=128,freeze_backbone=True):
        super().__init__()
        try:
            import torchaudio
        except ImportError as exc:
            raise ImportError("install the 'advanced' extra to use Wav2Vec2AudioEncoder") from exc
        self.bundle=torchaudio.pipelines.WAV2VEC2_BASE
        self.backbone=self.bundle.get_model()
        if freeze_backbone:
            for p in self.backbone.parameters(): p.requires_grad=False
        self.projection=nn.Linear(768,output_dim)

    def forward(self,waveforms):
        if waveforms.ndim==3: waveforms=waveforms.squeeze(1)
        features,_=self.backbone.extract_features(waveforms)
        return self.projection(features[-1].mean(dim=1))
