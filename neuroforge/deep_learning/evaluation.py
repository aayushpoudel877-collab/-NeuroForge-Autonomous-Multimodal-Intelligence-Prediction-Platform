from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from .metrics import binary_metrics

@torch.no_grad()
def evaluate_model(model,batches,device="cpu",ablation=None):
    model.eval(); logits=[]; labels=[]
    for batch in batches:
        text=torch.as_tensor(batch["text"],device=device,dtype=torch.long)
        image=torch.as_tensor(batch["image"],device=device,dtype=torch.float32)
        audio=torch.as_tensor(batch["audio"],device=device,dtype=torch.float32)
        series=torch.as_tensor(batch["series"],device=device,dtype=torch.float32)
        if ablation=="text": text=torch.zeros_like(text)
        elif ablation=="image": image=torch.zeros_like(image)
        elif ablation=="audio": audio=torch.zeros_like(audio)
        elif ablation=="temporal": series=torch.zeros_like(series)
        logits.append(model(text,image,audio,series).detach().cpu().numpy()); labels.append(np.asarray(batch["labels"]))
    if not labels:return {"samples":0,**binary_metrics([],[])}
    y=np.concatenate(labels); z=np.concatenate(logits)
    return {"samples":int(len(y)),**binary_metrics(z,y)}

def save_evaluation(report,path):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(report,indent=2))
