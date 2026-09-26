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
    return {"sample_ids":[r.sample_id for r in records],"labels":np.asarray([r.label for r in records],dtype=np.int64)}
