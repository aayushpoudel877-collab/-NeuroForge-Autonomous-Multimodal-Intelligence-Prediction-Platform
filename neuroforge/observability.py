from __future__ import annotations
from dataclasses import dataclass,asdict,field
from datetime import datetime,timezone
from pathlib import Path
import json,uuid

@dataclass
class InferenceTrace:
    request_id:str
    started_at:str
    finished_at:str
    status:str
    latency_ms:float
    modalities:list[str]=field(default_factory=list)
    prediction:str|None=None
    probability:float|None=None
    model_name:str|None=None
    model_version:str|None=None
    dataset_snapshot:str|None=None
    training_run:str|None=None
    artifact_sha256:str|None=None
    error_type:str|None=None

class TraceStore:
    def __init__(self,path="logs/inference.jsonl"):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
    def append(self,trace:InferenceTrace):
        with self.path.open("a",encoding="utf-8") as f:
            f.write(json.dumps(asdict(trace),sort_keys=True)+"\n")
    def list(self,limit=100):
        if limit<1: raise ValueError("limit must be positive")
        if not self.path.exists(): return []
        rows=[json.loads(x) for x in self.path.read_text(encoding="utf-8").splitlines() if x.strip()]
        return rows[-limit:]

def new_request_id():
    return uuid.uuid4().hex

def utc_now():
    return datetime.now(timezone.utc).isoformat()
