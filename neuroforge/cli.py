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
from .deep_learning.research import ExperimentSpec,ResearchExperimentEngine,save_research_report,modality_ablation_matrix
from .model_card import build_model_card,save_model_card
from .deep_learning.runs import RunConfig
from .deep_learning.training_session import train_model,seed_everything

def main():
    parser=argparse.ArgumentParser(prog="neuroforge")
    sub=parser.add_subparsers(dest="command",required=True)
    s=sub.add_parser("prepare-demo"); s.add_argument("--output",default="data/synthetic"); s.add_argument("--samples",type=int,default=24)
    for name in ("evaluate","benchmark","robustness"):
        p=sub.add_parser(name); p.add_argument("--manifest",required=True); p.add_argument("--root",default="."); p.add_argument("--checkpoint"); p.add_argument("--output",default=f"reports/{name}.json")
    sub.choices["evaluate"].add_argument("--calibrate",action="store_true")
    e=sub.add_parser("experiment"); e.add_argument("--manifest",required=True); e.add_argument("--root",default=".")
    e.add_argument("--epochs",type=int,default=3); e.add_argument("--learning-rates",default="0.001")
    e.add_argument("--batch-sizes",default="4"); e.add_argument("--seeds",default="42,43,44")
    e.add_argument("--checkpoint-dir",default="models/checkpoints/research"); e.add_argument("--output",default="reports/research-experiment.json")
    t=sub.add_parser("train"); t.add_argument("--manifest",required=True); t.add_argument("--root",default="."); t.add_argument("--epochs",type=int,default=3)
    t.add_argument("--batch-size",type=int,default=4); t.add_argument("--learning-rate",type=float,default=1e-3); t.add_argument("--checkpoint-dir",default="models/checkpoints")
    args=parser.parse_args()
    if args.command=="prepare-demo":
        m=write_synthetic_dataset(args.output,args.samples); print(f"wrote {len(m.records)} samples to {args.output}"); return
    dataset=ManifestDataset(DatasetManifest.load_json(args.manifest),args.root)
    if args.command=="experiment":
        train_manifest,val_manifest,test_manifest=dataset.manifest.split(train=.7,val=.15,seed=42)
        train_ds=ManifestDataset(train_manifest,args.root)
        val_ds=ManifestDataset(val_manifest,args.root)
        test_ds=ManifestDataset(test_manifest,args.root)
        batch_size=min(int(x) for x in args.batch_sizes.split(","))
        test_batches=batch_dataset(test_ds,batch_size)
        def model_factory():
            return NeuroForgeMultimodalModel()
        def train_batches_factory(config):
            return batch_dataset(train_ds,config.batch_size)
        def val_batches_factory(config):
            return batch_dataset(val_ds,config.batch_size)
        spec=ExperimentSpec(
            learning_rates=tuple(float(x) for x in args.learning_rates.split(",") if x.strip()),
            batch_sizes=tuple(int(x) for x in args.batch_sizes.split(",") if x.strip()),
            epochs=args.epochs,
            seeds=tuple(int(x) for x in args.seeds.split(",") if x.strip()),
        )
        engine=ResearchExperimentEngine(model_factory,train_batches_factory,val_batches_factory,test_batches,args.checkpoint_dir)
        report=engine.run(spec)
        if report.get("best_run"):
            best_checkpoint=report["best_run"]["checkpoint"]
            seed_everything(int(report["best_run"]["seed"]))
            best_model=NeuroForgeMultimodalModel()
            state=torch.load(best_checkpoint,map_location="cpu",weights_only=False)
            best_model.load_state_dict(state["model"])
            report["best_run"]["modality_ablation"]=modality_ablation_matrix(best_model,test_batches)
            card=build_model_card(
                "NeuroForgeMultimodalModel",
                "binary multimodal prediction",
                report["best_run"]["calibrated"],
                ("text","image","audio","temporal"),
                ["Synthetic data is a development fixture; results are not production evidence."],
            )
            report["model_card"]=card
            save_model_card(card,"reports/research-model-card.json")
        save_research_report(report,args.output)
        print(json.dumps(report,indent=2)); return
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
    elif args.calibrate:
        _,calibration_manifest,test_manifest=dataset.manifest.split(train=.7,val=.15,seed=42)
        calibration_batches=batch_dataset(ManifestDataset(calibration_manifest,args.root))
        test_batches=batch_dataset(ManifestDataset(test_manifest,args.root))
        report=evaluate_calibrated(model,test_batches,calibration_batches=calibration_batches)
    else: report=evaluate_model(model,batches)
    save_evaluation(report,args.output); print(json.dumps(report,indent=2))

if __name__=="__main__": main()
