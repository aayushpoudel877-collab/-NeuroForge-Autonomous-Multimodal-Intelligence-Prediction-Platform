from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from .manifest import DatasetManifest,SampleRecord
from .preprocessors import tokenize_text,prepare_image,prepare_audio,prepare_series

@dataclass
class PreparedSample:
    sample_id:str
    label:int
    text:np.ndarray
    image:np.ndarray
    audio:np.ndarray
    series:np.ndarray

class ManifestDataset:
    """Loads a manifest and converts samples to the neural model tensor contract."""
    def __init__(self,manifest,root="."):
        self.manifest=manifest if isinstance(manifest,DatasetManifest) else DatasetManifest(manifest)
        self.root=Path(root)
    def _load(self,path,kind):
        if not path: raise ValueError(f"missing {kind} path for dense multimodal training")
        return np.load(self.root/path)
    def __len__(self): return len(self.manifest.records)
    def __getitem__(self,index):
        r=self.manifest.records[index]
        return PreparedSample(r.sample_id,r.label,tokenize_text(r.text),
            prepare_image(self._load(r.image_path,"image")),
            prepare_audio(self._load(r.audio_path,"audio")),
            prepare_series(self._load(r.series_path,"series")))

def write_synthetic_dataset(directory="data/synthetic",samples=24,seed=42):
    rng=np.random.default_rng(seed); root=Path(directory); root.mkdir(parents=True,exist_ok=True)
    records=[]
    for i in range(samples):
        label=i%2; stem=f"sample_{i:04d}"
        np.save(root/f"{stem}_image.npy",rng.normal(.25+.35*label,.12,(64,64,3)).clip(0,1))
        np.save(root/f"{stem}_audio.npy",rng.normal(.1+.4*label,.2,16000).astype(np.float32))
        np.save(root/f"{stem}_series.npy",(rng.normal(0,1,(64,1))+label*.7).astype(np.float32))
        records.append(SampleRecord(stem,label,f"sample class {label}",f"{stem}_image.npy",f"{stem}_audio.npy",f"{stem}_series.npy"))
    manifest=DatasetManifest(records); manifest.save_json(root/"manifest.json"); return manifest
