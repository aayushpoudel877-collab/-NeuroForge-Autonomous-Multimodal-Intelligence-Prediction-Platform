from __future__ import annotations
from dataclasses import dataclass,field
from collections import deque
import math

@dataclass
class PredictionMonitor:
    window_size:int=100
    probabilities:deque=field(default_factory=lambda:deque(maxlen=100))
    latencies_ms:deque=field(default_factory=lambda:deque(maxlen=100))
    def record(self,probability:float,latency_ms:float):
        self.probabilities.append(float(probability));self.latencies_ms.append(float(latency_ms))
    def snapshot(self):
        p=list(self.probabilities);l=list(self.latencies_ms)
        entropy=0.0
        if p:
            entropy=sum(-(x*math.log(max(x,1e-9))+(1-x)*math.log(max(1-x,1e-9))) for x in p)/len(p)
        return {"samples":len(p),"mean_probability":sum(p)/len(p) if p else 0.0,"mean_latency_ms":sum(l)/len(l) if l else 0.0,"prediction_entropy":entropy}
