from dataclasses import dataclass
import numpy as np
@dataclass
class FusionOutput:
    probability:float
    confidence:float
    weights:dict[str,float]
class ConfidenceWeightedFusion:
    def fuse(self,results):
        if not results:raise ValueError("No modality results")
        names=[r[0] for r in results];scores=np.array([r[1] for r in results]);conf=np.maximum([r[2] for r in results],1e-6)
        w=np.exp(np.log(conf)-np.max(np.log(conf)));w=w/w.sum();p=float((w*scores).sum())
        c=float(np.clip(.65*(w*conf).sum()+.35*(1-scores.std()),0,.999))
        return FusionOutput(p,c,dict(zip(names,w.tolist())))
