from __future__ import annotations
import argparse,json
import torch
from .data.manifest import DatasetManifest
from .data.datasets import ManifestDataset,write_synthetic_dataset
from .deep_learning.collate import batch_dataset
from .deep_learning.unified_model import NeuroForgeMultimodalModel
from .deep_learning.evaluation import evaluate_model,save_evaluation
from .deep_learning.benchmark import benchmark_modalities

def main():
    parser=argparse.ArgumentParser(prog="neuroforge")
    sub=parser.add_subparsers(dest="command",required=True)
    s=sub.add_parser("prepare-demo"); s.add_argument("--output",default="data/synthetic"); s.add_argument("--samples",type=int,default=24)
    for name in ("evaluate","benchmark"):
        p=sub.add_parser(name); p.add_argument("--manifest",required=True); p.add_argument("--root",default="."); p.add_argument("--checkpoint"); p.add_argument("--output",default=f"reports/{name}.json")
    args=parser.parse_args()
    if args.command=="prepare-demo":
        m=write_synthetic_dataset(args.output,args.samples); print(f"wrote {len(m.records)} samples to {args.output}"); return
    dataset=ManifestDataset(DatasetManifest.load_json(args.manifest),args.root)
    model=NeuroForgeMultimodalModel()
    if args.checkpoint: model.load_state_dict(torch.load(args.checkpoint,map_location="cpu"))
    batches=batch_dataset(dataset)
    report=benchmark_modalities(model,batches) if args.command=="benchmark" else evaluate_model(model,batches)
    save_evaluation(report,args.output); print(json.dumps(report,indent=2))

if __name__=="__main__": main()
