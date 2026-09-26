from __future__ import annotations
from dataclasses import dataclass,field
from time import time

@dataclass
class ApiMetrics:
    started_at:float=field(default_factory=time)
    predictions:int=0
    failures:int=0
    batches:int=0

    def prediction(self,failed:bool=False):
        self.predictions+=1
        if failed:self.failures+=1

    def batch(self):
        self.batches+=1

    def snapshot(self):
        total=self.predictions
        return {
            "uptime_seconds":time()-self.started_at,
            "predictions_total":total,
            "prediction_failures_total":self.failures,
            "batches_total":self.batches,
            "failure_rate":self.failures/total if total else 0.0,
        }
