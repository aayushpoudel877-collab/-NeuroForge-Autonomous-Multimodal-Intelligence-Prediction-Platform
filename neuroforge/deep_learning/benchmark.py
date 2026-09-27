from __future__ import annotations
from .evaluation import evaluate_model
ABLATIONS=("full","text","image","audio","temporal")

def benchmark_modalities(model,batches,device="cpu"):
    return {mode:evaluate_model(model,batches,device,None if mode=="full" else mode) for mode in ABLATIONS}
