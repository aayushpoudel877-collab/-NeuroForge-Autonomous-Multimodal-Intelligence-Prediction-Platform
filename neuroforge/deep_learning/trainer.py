from __future__ import annotations
import random
import numpy as np
import torch
from torch import nn
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score
from .fusion_net import MultimodalFusionNet

class SyntheticMultimodalTrainer:
    """Reproducible trainer for validating the neural architecture without private data."""
    def __init__(self,seed:int=42):
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    def make_data(self,n:int=512):
        rng=np.random.default_rng(42)
        dims={"text":8,"image":10,"audio":8,"timeseries":6}
        x={k:rng.normal(size=(n,d)).astype("float32") for k,d in dims.items()}
        signal=.7*x["text"][:,0]+.5*x["image"][:,0]-.4*x["audio"][:,1]+.6*x["timeseries"][:,0]
        return {k:torch.from_numpy(v) for k,v in x.items()},torch.from_numpy((signal>0).astype("float32"))
    def fit(self,epochs:int=8,lr:float=2e-3):
        features,y=self.make_data(); model=MultimodalFusionNet({"text":8,"image":10,"audio":8,"timeseries":6})
        opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4); loss_fn=nn.BCEWithLogitsLoss()
        model.train()
        for _ in range(epochs):
            opt.zero_grad(); logits,_=model(features); loss=loss_fn(logits,y); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad(): p=torch.sigmoid(model(features)[0]).numpy()
        pred=(p>=.5).astype(int); truth=y.numpy().astype(int)
        return model,{"accuracy":accuracy_score(truth,pred),"f1":f1_score(truth,pred),"roc_auc":roc_auc_score(truth,p)}
