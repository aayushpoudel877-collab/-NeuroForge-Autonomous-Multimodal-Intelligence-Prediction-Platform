from __future__ import annotations
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score

def binary_metrics(logits,labels,threshold=.5):
    z=np.asarray(logits,dtype=float).reshape(-1)
    y=np.asarray(labels,dtype=int).reshape(-1)
    if len(y)==0:return {"accuracy":0.0,"precision":0.0,"recall":0.0,"f1":0.0,"auroc":0.0,"pr_auc":0.0,"brier":0.0,"ece":0.0}
    probabilities=1/(1+np.exp(-np.clip(z,-40,40)))
    pred=(probabilities>=threshold).astype(int)
    accuracy=float((pred==y).mean())
    tp=int(((pred==1)&(y==1)).sum()); fp=int(((pred==1)&(y==0)).sum()); fn=int(((pred==0)&(y==1)).sum())
    precision=tp/(tp+fp) if tp+fp else 0.0
    recall=tp/(tp+fn) if tp+fn else 0.0
    f1=2*precision*recall/(precision+recall) if precision+recall else 0.0
    auroc=float(roc_auc_score(y,probabilities)) if len(np.unique(y))==2 else 0.0
    pr_auc=float(average_precision_score(y,probabilities)) if len(np.unique(y))==2 else 0.0
    brier=float(np.mean((probabilities-y)**2))
    bins=np.linspace(0,1,11); ece=0.0
    for lo,hi in zip(bins[:-1],bins[1:]):
        mask=(probabilities>=lo)&((probabilities<hi) if hi<1 else (probabilities<=hi))
        if mask.any(): ece+=float(mask.mean())*abs(float(probabilities[mask].mean())-float(y[mask].mean()))
    return {"accuracy":accuracy,"precision":precision,"recall":recall,"f1":f1,
            "auroc":auroc,"pr_auc":pr_auc,"brier":brier,"ece":float(ece)}
