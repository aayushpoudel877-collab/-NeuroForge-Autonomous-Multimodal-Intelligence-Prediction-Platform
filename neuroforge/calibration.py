from __future__ import annotations
import numpy as np

class TemperatureScaler:
    """Temperature scaling for binary probabilities or logits."""
    def __init__(self,temperature:float=1.0):
        self.temperature=max(float(temperature),1e-3)

    def transform(self,probabilities):
        p=np.clip(np.asarray(probabilities,dtype=float),1e-6,1-1e-6)
        logits=np.log(p/(1-p))/self.temperature
        return 1/(1+np.exp(-logits))

    def transform_logits(self,logits):
        z=np.asarray(logits,dtype=float)
        return 1/(1+np.exp(-np.clip(z/self.temperature,-40,40)))
