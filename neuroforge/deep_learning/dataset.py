from __future__ import annotations
from pathlib import Path
import numpy as np

class MultimodalDataset:
    """Dependency-light manifest dataset; production decoders can be injected."""
    def __init__(self,records,text_encoder=None,array_loader=None):
        self.records=list(records)
        self.text_encoder=text_encoder or self._default_text_encoder
        self.array_loader=array_loader or (lambda path: np.load(path))

    @staticmethod
    def _default_text_encoder(text):
        tokens=np.frombuffer(text.encode(),dtype=np.uint8)[:128]
        return np.pad(tokens,(0,128-len(tokens)))

    def __len__(self): return len(self.records)

    def __getitem__(self,index):
        r=self.records[index]
        return {
            "sample_id":r.sample_id,
            "label":np.int64(r.label),
            "text":self.text_encoder(r.text),
            "image":self.array_loader(r.image_path) if r.image_path else None,
            "audio":self.array_loader(r.audio_path) if r.audio_path else None,
            "series":self.array_loader(r.series_path) if r.series_path else None,
        }
