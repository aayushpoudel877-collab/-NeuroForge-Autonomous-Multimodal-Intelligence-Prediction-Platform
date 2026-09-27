from __future__ import annotations
from dataclasses import dataclass

@dataclass
class EpochResult:
    loss:float
    metrics:dict

class TrainerEngine:
    def __init__(self,model,optimizer,criterion,device="cpu"):
        self.model=model; self.optimizer=optimizer; self.criterion=criterion; self.device=device
    def train_epoch(self,batches):
        self.model.train(); total=0.0; count=0
        for batch in batches:
            self.optimizer.zero_grad(set_to_none=True)
            logits=self.model(batch["text"].to(self.device),batch["image"].to(self.device),batch["audio"].to(self.device),batch["series"].to(self.device))
            loss=self.criterion(logits,batch["labels"].float().to(self.device))
            loss.backward(); self.optimizer.step()
            n=batch["labels"].shape[0]; total+=float(loss.detach())*n; count+=n
        return total/count if count else 0.0
