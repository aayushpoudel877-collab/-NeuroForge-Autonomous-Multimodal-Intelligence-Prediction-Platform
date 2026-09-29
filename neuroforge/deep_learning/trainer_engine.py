from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from .metrics import binary_metrics

@dataclass
class EpochResult:
    loss:float
    metrics:dict

class TrainerEngine:
    def __init__(self,model,optimizer,criterion,device="cpu"):
        self.model=model; self.optimizer=optimizer; self.criterion=criterion; self.device=device

    def _tensors(self,batch):
        return (torch.as_tensor(batch["text"],device=self.device,dtype=torch.long),
                torch.as_tensor(batch["image"],device=self.device,dtype=torch.float32),
                torch.as_tensor(batch["audio"],device=self.device,dtype=torch.float32),
                torch.as_tensor(batch["series"],device=self.device,dtype=torch.float32),
                torch.as_tensor(batch["labels"],device=self.device,dtype=torch.float32),
                torch.as_tensor(batch["modality_mask"],device=self.device,dtype=torch.bool)
                if "modality_mask" in batch else None)

    def _forward(self,batch):
        text,image,audio,series,y,mask=self._tensors(batch)
        z=self.model(text,image,audio,series,mask) if mask is not None else self.model(text,image,audio,series)
        return z,y

    def train_epoch(self,batches):
        self.model.train(); total=0.0; count=0; logits=[]; labels=[]
        for batch in batches:
            z,y=self._forward(batch); self.optimizer.zero_grad(set_to_none=True)
            loss=self.criterion(z,y); loss.backward(); self.optimizer.step()
            n=y.shape[0]; total+=float(loss.detach())*n; count+=n
            logits.append(z.detach().cpu().numpy()); labels.append(y.detach().cpu().numpy())
        if not count:return EpochResult(0.0,binary_metrics([],[]))
        return EpochResult(total/count,binary_metrics(np.concatenate(logits),np.concatenate(labels)))

    @torch.no_grad()
    def validate(self,batches):
        self.model.eval(); total=0.0; count=0; logits=[]; labels=[]
        for batch in batches:
            z,y=self._forward(batch); n=y.shape[0]; total+=float(self.criterion(z,y))*n; count+=n
            logits.append(z.cpu().numpy()); labels.append(y.cpu().numpy())
        if not count:return EpochResult(0.0,binary_metrics([],[]))
        return EpochResult(total/count,binary_metrics(np.concatenate(logits),np.concatenate(labels)))
