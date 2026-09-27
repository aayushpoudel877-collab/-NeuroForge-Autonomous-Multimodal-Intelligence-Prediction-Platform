from __future__ import annotations
from dataclasses import dataclass,asdict
from datetime import datetime,timezone
from pathlib import Path
import hashlib,json

@dataclass
class Artifact:
    name:str
    version:str
    path:str
    sha256:str
    metrics:dict
    created_at:str

def hash_file(path):
    digest=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): digest.update(chunk)
    return digest.hexdigest()

class ArtifactStore:
    def __init__(self,root="models/artifacts"):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
    def register_file(self,name,version,path,metrics=None):
        source=Path(path)
        if not source.exists(): raise FileNotFoundError(path)
        target=self.root/f"{name}-{version}{source.suffix}"
        target.write_bytes(source.read_bytes())
        artifact=Artifact(name,version,str(target),hash_file(target),metrics or {},datetime.now(timezone.utc).isoformat())
        (self.root/f"{name}-{version}.json").write_text(json.dumps(asdict(artifact),indent=2))
        return artifact
    def get(self,name,version):
        metadata=self.root/f"{name}-{version}.json"
        if not metadata.exists(): raise KeyError(f"{name}:{version}")
        return Artifact(**json.loads(metadata.read_text()))

class PromotionGate:
    def __init__(self,metric="f1",minimum=.75):
        self.metric=metric;self.minimum=minimum
    def evaluate(self,metrics):
        value=float(metrics.get(self.metric,-1))
        return {"approved":value>=self.minimum,"metric":self.metric,"value":value,"minimum":self.minimum}

class LifecycleHistory:
    def __init__(self,path="models/lifecycle.json"):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
    def record(self,event):
        data=json.loads(self.path.read_text()) if self.path.exists() else []
        data.append(event);self.path.write_text(json.dumps(data,indent=2))
    def list(self): return json.loads(self.path.read_text()) if self.path.exists() else []
