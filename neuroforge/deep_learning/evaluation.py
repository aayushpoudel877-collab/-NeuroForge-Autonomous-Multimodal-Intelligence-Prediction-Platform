from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from .metrics import binary_metrics

@torch.no_grad()
def evaluate_model(model,batches,device="cpu",ablation=None):
    model.eval(); logits=[]; labels=[]
    indices={"text":0,"image":1,"audio":2,"temporal":3}
    for batch in batches:
        text=torch.as_tensor(batch["text"],device=device,dtype=torch.long)
        image=torch.as_tensor(batch["image"],device=device,dtype=torch.float32)
        audio=torch.as_tensor(batch["audio"],device=device,dtype=torch.float32)
        series=torch.as_tensor(batch["series"],device=device,dtype=torch.float32)
        mask=torch.as_tensor(batch.get("modality_mask",np.ones((len(batch["labels"]),4),bool)),device=device,dtype=torch.bool)
        if ablation in indices: mask[:,indices[ablation]]=False
        logits.append(model(text,image,audio,series,mask).cpu().numpy()); labels.append(np.asarray(batch["labels"]))
    if not labels:return {"samples":0,**binary_metrics([],[])}
    y=np.concatenate(labels); z=np.concatenate(logits)
    return {"samples":int(len(y)),**binary_metrics(z,y)}

def save_evaluation(report,path):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(report,indent=2))
