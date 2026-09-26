from __future__ import annotations
from dataclasses import dataclass,asdict
from pathlib import Path
import csv,json,random

@dataclass
class SampleRecord:
    sample_id:str
    label:int
    text:str=""
    image_path:str=""
    audio_path:str=""
    series_path:str=""

class DatasetManifest:
    def __init__(self,records=None):
        self.records=list(records or [])
    def add(self,record:SampleRecord):
        self.records.append(record)
    def split(self,train=.8,val=.1,seed=42):
        if train<=0 or val<0 or train+val>=1: raise ValueError("invalid split")
        items=self.records.copy(); random.Random(seed).shuffle(items)
        n=len(items); a=int(n*train); b=a+int(n*val)
        return DatasetManifest(items[:a]),DatasetManifest(items[a:b]),DatasetManifest(items[b:])
    def save_json(self,path):
        p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps([asdict(x) for x in self.records],indent=2))
    @classmethod
    def load_json(cls,path):
        data=json.loads(Path(path).read_text())
        return cls(SampleRecord(**x) for x in data)
    def save_csv(self,path):
        p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
        with p.open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(asdict(self.records[0]).keys()) if self.records else ["sample_id","label","text","image_path","audio_path","series_path"])
            writer.writeheader();writer.writerows(asdict(x) for x in self.records)
