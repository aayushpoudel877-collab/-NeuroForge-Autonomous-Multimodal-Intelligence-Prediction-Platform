from __future__ import annotations
from collections import Counter
from .manifest import DatasetManifest

def profile_manifest(manifest:DatasetManifest):
    labels=Counter(r.label for r in manifest.records)
    missing={
        "text":sum(not r.text for r in manifest.records),
        "image":sum(not r.image_path for r in manifest.records),
        "audio":sum(not r.audio_path for r in manifest.records),
        "series":sum(not r.series_path for r in manifest.records),
    }
    return {"samples":len(manifest.records),"label_counts":dict(labels),"missing_modalities":missing}

def validate_manifest(manifest:DatasetManifest):
    ids=[r.sample_id for r in manifest.records]
    errors=[]
    if len(ids)!=len(set(ids)): errors.append("duplicate sample_id values")
    if any(r.label not in (0,1) for r in manifest.records): errors.append("labels must be 0 or 1")
    return errors
