from __future__ import annotations
import numpy as np
from ..calibration import TemperatureScaler

def fit_temperature(logits, labels, steps=100, learning_rate=.05):
    """Fit a scalar temperature by minimizing binary cross-entropy on logits."""
    z=np.asarray(logits,dtype=float).reshape(-1)
    y=np.asarray(labels,dtype=float).reshape(-1)
    if z.size==0 or z.size!=y.size:
        return TemperatureScaler(1.0)
    if not np.isfinite(z).all() or not np.isfinite(y).all():
        raise ValueError("logits and labels must be finite")
    temperature=1.0
    for _ in range(max(1,int(steps))):
        scaled=np.clip(z/temperature,-40,40)
        p=1/(1+np.exp(-scaled))
        grad=float(np.mean((p-y)*(-z/(temperature**2))))
        temperature=max(.05,min(20.0,temperature-learning_rate*grad))
    return TemperatureScaler(temperature)
