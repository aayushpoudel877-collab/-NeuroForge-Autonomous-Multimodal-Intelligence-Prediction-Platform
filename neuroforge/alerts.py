from __future__ import annotations
from dataclasses import dataclass,asdict
from datetime import datetime,timezone
from pathlib import Path
import json

@dataclass
class Alert:
    alert_type:str
    severity:str
    message:str
    value:float
    threshold:float
    created_at:str

class AlertStore:
    def __init__(self,path="logs/alerts.jsonl"):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
    def emit(self,alert:Alert):
        with self.path.open("a",encoding="utf-8") as f:
            f.write(json.dumps(asdict(alert),sort_keys=True)+"\n")
    def recent(self,limit=100):
        if limit<1: raise ValueError("limit must be positive")
        if not self.path.exists(): return []
        rows=[json.loads(x) for x in self.path.read_text(encoding="utf-8").splitlines() if x.strip()]
        return rows[-limit:]

def threshold_alert(alert_type,value,threshold,message,severity="warning"):
    if value<threshold: return None
    return Alert(alert_type,severity,message,float(value),float(threshold),datetime.now(timezone.utc).isoformat())
