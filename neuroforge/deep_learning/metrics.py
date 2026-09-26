from __future__ import annotations
import numpy as np

def binary_metrics(logits,labels,threshold=.5):
    probabilities=1/(1+np.exp(-np.asarray(logits,dtype=float)))
    y=np.asarray(labels,dtype=int)
    pred=(probabilities>=threshold).astype(int)
    accuracy=float((pred==y).mean()) if len(y) else 0.0
    tp=int(((pred==1)&(y==1)).sum());fp=int(((pred==1)&(y==0)).sum());fn=int(((pred==0)&(y==1)).sum())
    precision=tp/(tp+fp) if tp+fp else 0.0
    recall=tp/(tp+fn) if tp+fn else 0.0
    f1=2*precision*recall/(precision+recall) if precision+recall else 0.0
    return {"accuracy":accuracy,"precision":precision,"recall":recall,"f1":f1}
