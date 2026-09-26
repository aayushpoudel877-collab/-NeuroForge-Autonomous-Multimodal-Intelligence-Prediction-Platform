from __future__ import annotations
from dataclasses import dataclass,asdict
from pathlib import Path
import json

@dataclass
class RegisteredModel:
    name:str
    version:str
    artifact:str
    metrics:dict
    stage:str="candidate"

class JsonModelRegistry:
    def __init__(self,path:str="models/registry.json"):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
    def _load(self):
        return json.loads(self.path.read_text()) if self.path.exists() else []
    def register(self,model:RegisteredModel):
        items=[x for x in self._load() if not (x["name"]==model.name and x["version"]==model.version)]
        items.append(asdict(model));self.path.write_text(json.dumps(items,indent=2))
    def promote(self,name:str,version:str,stage:str="production"):
        items=self._load();found=False
        for x in items:
            if x["name"]==name and x["version"]==version:x["stage"]=stage;found=True
        if not found:raise KeyError(f"{name}:{version}")
        self.path.write_text(json.dumps(items,indent=2))
    def list(self): return self._load()
