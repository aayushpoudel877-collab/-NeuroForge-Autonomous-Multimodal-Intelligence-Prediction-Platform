from __future__ import annotations
import numpy as np
from .evaluation import evaluate_model

def random_mask_batches(batches,drop_probability=.25,seed=42):
    if not 0<=drop_probability<1: raise ValueError("drop_probability must be in [0,1)")
    rng=np.random.default_rng(seed); masked=[]
    for batch in batches:
        clone=dict(batch)
        existing=np.asarray(batch.get("modality_mask",np.ones((len(batch["labels"]),4),bool)),dtype=bool).copy()
        drops=rng.random(existing.shape)<drop_probability
        candidate=existing & ~drops
        for row in np.where(~candidate.any(axis=1))[0]: candidate[row,rng.integers(0,4)]=True
        clone["modality_mask"]=candidate
        masked.append(clone)
    return masked

def robustness_report(model,batches,drop_probabilities=(0.0,.1,.25,.5),device="cpu",seed=42):
    return {str(p):evaluate_model(model,random_mask_batches(batches,p,seed),device) for p in drop_probabilities}
