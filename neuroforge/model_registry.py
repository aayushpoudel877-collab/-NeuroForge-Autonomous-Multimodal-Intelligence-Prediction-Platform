from __future__ import annotations
from dataclasses import dataclass,asdict
from datetime import datetime,timezone

@dataclass(frozen=True)
class ModelArtifact:
    name:str
    version:str
    stage:str="candidate"
    metric:float|None=None
    created_at:str=""
    def __post_init__(self):
        if not self.created_at: object.__setattr__(self,"created_at",datetime.now(timezone.utc).isoformat())

class InMemoryModelRegistry:
    def __init__(self): self._items={}
    def register(self,artifact:ModelArtifact): self._items[(artifact.name,artifact.version)]=artifact
    def promote(self,name:str,version:str,stage:str="production"):
        key=(name,version)
        if key not in self._items: raise KeyError(key)
        old=self._items[key]; self._items[key]=ModelArtifact(old.name,old.version,stage,old.metric,old.created_at)
    def get(self,name:str,version:str): return self._items.get((name,version))
    def list(self): return [asdict(x) for x in self._items.values()]
