from __future__ import annotations
from dataclasses import dataclass
import torch

@dataclass
class EpochResult:
    loss:float
    metrics:dict

class TrainerEngine:
    def __init__(self,model,optimizer,criterion,device="cpu"):
        self.model=model;self.optimizer=optimizer;self.criterion=criterion;self.device=device
    def train_epoch(self,batches):
        self.model.train();total=0.0;count=0
        for batch in batches:
            self.optimizer.zero_grad(set_to_none=True)
            text=torch.as_tensor(batch["text"],device=self.device,dtype=torch.long)
            image=torch.as_tensor(batch["image"],device=self.device,dtype=torch.float32)
            audio=torch.as_tensor(batch["audio"],device=self.device,dtype=torch.float32)
            series=torch.as_tensor(batch["series"],device=self.device,dtype=torch.float32)
            labels=torch.as_tensor(batch["labels"],device=self.device,dtype=torch.float32)
            logits=self.model(text,image,audio,series)
            loss=self.criterion(logits,labels);loss.backward();self.optimizer.step()
            n=labels.shape[0];total+=float(loss.detach())*n;count+=n
        return total/count if count else 0.0
