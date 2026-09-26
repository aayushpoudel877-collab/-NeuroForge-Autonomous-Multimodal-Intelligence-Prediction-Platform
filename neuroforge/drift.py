from __future__ import annotations
import numpy as np

class DriftDetector:
    def __init__(self,threshold:float=0.15): self.threshold=threshold
    def score(self,reference,current):
        a=np.asarray(reference,dtype=float);b=np.asarray(current,dtype=float)
        if len(a)==0 or len(b)==0:return 0.0
        bins=np.linspace(min(a.min(),b.min()),max(a.max(),b.max())+1e-9,11)
        pa,_=np.histogram(a,bins=bins);pb,_=np.histogram(b,bins=bins)
        pa=(pa+1)/(pa.sum()+len(pa));pb=(pb+1)/(pb.sum()+len(pb))
        psi=float(np.sum((pb-pa)*np.log(pb/pa)))
        return psi
    def is_drifted(self,reference,current): return self.score(reference,current)>self.threshold
