from __future__ import annotations
from dataclasses import dataclass,asdict
from pathlib import Path
from datetime import datetime,timezone
import json

@dataclass
class Experiment:
    run_id:str
    model:str
    parameters:dict
    metrics:dict
    created_at:str

class ExperimentTracker:
    def __init__(self,path="artifacts/experiments.jsonl"):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
    def log(self,run_id,model,parameters,metrics):
        item=Experiment(run_id,model,parameters,metrics,datetime.now(timezone.utc).isoformat())
        with self.path.open("a") as f:f.write(json.dumps(asdict(item))+"\n")
        return item
    def runs(self):
        if not self.path.exists():return []
        return [json.loads(x) for x in self.path.read_text().splitlines() if x.strip()]
