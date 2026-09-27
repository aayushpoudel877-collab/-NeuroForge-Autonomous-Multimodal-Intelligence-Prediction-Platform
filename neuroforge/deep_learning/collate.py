from __future__ import annotations
import numpy as np

def collate_samples(samples):
    if not samples: raise ValueError("cannot collate an empty batch")
    return {"sample_id":[x.sample_id for x in samples],
        "labels":np.asarray([x.label for x in samples],dtype=np.float32),
        "text":np.stack([x.text for x in samples]),"image":np.stack([x.image for x in samples]),
        "audio":np.stack([x.audio for x in samples]),"series":np.stack([x.series for x in samples])}

def batch_dataset(dataset,batch_size=8):
    if batch_size<1: raise ValueError("batch_size must be positive")
    return [collate_samples([dataset[i] for i in range(start,min(start+batch_size,len(dataset)))])
            for start in range(0,len(dataset),batch_size)]
