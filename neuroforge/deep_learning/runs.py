from __future__ import annotations
from dataclasses import dataclass,asdict
from datetime import datetime,timezone
from pathlib import Path
import hashlib,json

@dataclass(frozen=True)
class RunConfig:
    model_name:str="neuroforge_multimodal"
    learning_rate:float=1e-3
    batch_size:int=32
    epochs:int=10
    seed:int=42

def run_id(config:RunConfig)->str:
    payload=json.dumps(asdict(config),sort_keys=True,separators=(",",":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:12]

@dataclass
class TrainingRun:
    run_id:str
    config:dict
    status:str="created"
    metrics:dict|None=None
    started_at:str|None=None
    finished_at:str|None=None

    def start(self):
        self.status="running"
        self.started_at=datetime.now(timezone.utc).isoformat()
    def finish(self,metrics:dict,status="completed"):
        self.metrics=metrics
        self.status=status
        self.finished_at=datetime.now(timezone.utc).isoformat()

class RunStore:
    def __init__(self,path="models/runs.jsonl"):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
    def append(self,run:TrainingRun):
        with self.path.open("a",encoding="utf-8") as f:
            f.write(json.dumps(asdict(run),sort_keys=True)+"\n")
    def list(self):
        if not self.path.exists(): return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]
