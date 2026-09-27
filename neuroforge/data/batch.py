from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class Batch:
    text:np.ndarray
    image:np.ndarray
    audio:np.ndarray
    series:np.ndarray
    labels:np.ndarray

def collate_records(records):
    if not records: raise ValueError("records cannot be empty")
    required=("text","image","audio","series")
    if any(any(item.get(name) is None for name in required) for item in records):
        raise ValueError("all four modalities are required for the dense multimodal trainer")
    shapes={name:np.asarray(records[0][name]).shape for name in required}
    for item in records[1:]:
        for name in required:
            if np.asarray(item[name]).shape!=shapes[name]:
                raise ValueError(f"inconsistent {name} shape in batch")
    return {
        "sample_ids":[r["sample_id"] for r in records],
        "text":np.stack([np.asarray(r["text"]) for r in records]),
        "image":np.stack([np.asarray(r["image"]) for r in records]),
        "audio":np.stack([np.asarray(r["audio"]) for r in records]),
        "series":np.stack([np.asarray(r["series"]) for r in records]),
        "labels":np.asarray([r["label"] for r in records],dtype=np.int64),
    }
