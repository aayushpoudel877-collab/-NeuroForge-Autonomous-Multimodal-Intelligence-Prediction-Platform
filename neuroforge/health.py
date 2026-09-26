from __future__ import annotations
from dataclasses import dataclass
from time import monotonic

@dataclass
class HealthState:
    started_at:float
    requests:int=0
    errors:int=0

class ServiceHealth:
    def __init__(self):self.state=HealthState(monotonic())
    def request(self,error:bool=False):
        self.state.requests+=1
        if error:self.state.errors+=1
    def snapshot(self):
        uptime=monotonic()-self.state.started_at
        rate=self.state.errors/self.state.requests if self.state.requests else 0.0
        return {"status":"ok" if rate<.05 else "degraded","uptime_seconds":uptime,"requests":self.state.requests,"error_rate":rate}
