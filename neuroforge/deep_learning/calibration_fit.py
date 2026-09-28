from __future__ import annotations
import numpy as np
from .calibration import TemperatureScaler

def fit_temperature(logits,labels,steps=100,learning_rate=.05):
    z=np.asarray(logits,dtype=float); y=np.asarray(labels,dtype=float)
    if len(z)==0:return TemperatureScaler(1.0)
    temperature=1.0
    for _ in range(steps):
        p=1/(1+np.exp(-np.clip(z/temperature,-40,40)))
        grad=float(np.mean((p-y)*(-z/(temperature**2))))
        temperature=max(.05,min(20.0,temperature-learning_rate*grad))
    return TemperatureScaler(temperature)
