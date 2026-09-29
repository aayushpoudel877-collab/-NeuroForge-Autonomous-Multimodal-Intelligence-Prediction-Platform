from __future__ import annotations
import argparse,json
import torch
from .data.manifest import DatasetManifest
from .data.datasets import ManifestDataset,write_synthetic_dataset
from .deep_learning.collate import batch_dataset
from .deep_learning.unified_model import NeuroForgeMultimodalModel
from .deep_learning.evaluation import evaluate_model,save_evaluation,evaluate_calibrated
from .deep_learning.benchmark import benchmark_modalities
from .deep_learning.robustness import robustness_report
from .deep_learning.runs import RunConfig
from .deep_learning.training_session import train_model,seed_everything

def main():
    parser=argparse.ArgumentParser(prog="neuroforge")
    sub=parser.add_subparsers(dest="command",required=True)
    s=sub.add_parser("prepare-demo"); s.add_argument("--output",default="data/synthetic"); s.add_argument("--samples",type=int,default=24)
    for name in ("evaluate","benchmark","robustness"):
        p=sub.add_parser(name); p.add_argument("--manifest",required=True); p.add_argument("--root",default="."); p.add_argument("--checkpoint"); p.add_argument("--output",default=f"reports/{name}.json")
    sub.choices["evaluate"].add_argument("--calibrate",action="store_true")
    t=sub.add_parser("train"); t.add_argument("--manifest",required=True); t.add_argument("--root",default="."); t.add_argument("--epochs",type=int,default=3)
    t.add_argument("--batch-size",type=int,default=4); t.add_argument("--learning-rate",type=float,default=1e-3); t.add_argument("--checkpoint-dir",default="models/checkpoints")
    args=parser.parse_args()
    if args.command=="prepare-demo":
        m=write_synthetic_dataset(args.output,args.samples); print(f"wrote {len(m.records)} samples to {args.output}"); return
    dataset=ManifestDataset(DatasetManifest.load_json(args.manifest),args.root)
    if args.command=="train":
        train_manifest,val_manifest,_=dataset.manifest.split(train=.8,val=.1,seed=42)
        train_ds=ManifestDataset(train_manifest,args.root); val_ds=ManifestDataset(val_manifest,args.root)
        seed_everything(42); model=NeuroForgeMultimodalModel()
        config=RunConfig(learning_rate=args.learning_rate,batch_size=args.batch_size,epochs=args.epochs,seed=42)
        summary=train_model(model,batch_dataset(train_ds,args.batch_size),batch_dataset(val_ds,args.batch_size),config,args.checkpoint_dir)
        print(json.dumps({"run_id":summary.run_id,"best_epoch":summary.best_epoch,"best_f1":summary.best_metric,"checkpoint":summary.checkpoint},indent=2)); return
    seed_everything(42); model=NeuroForgeMultimodalModel()
    if args.checkpoint: model.load_state_dict(torch.load(args.checkpoint,map_location="cpu",weights_only=False))
    batches=batch_dataset(dataset)
    if args.command=="benchmark": report=benchmark_modalities(model,batches)
    elif args.command=="robustness": report=robustness_report(model,batches)
    elif args.calibrate: report=evaluate_calibrated(model,batches)
    else: report=evaluate_model(model,batches)
    save_evaluation(report,args.output); print(json.dumps(report,indent=2))

if __name__=="__main__": main()
