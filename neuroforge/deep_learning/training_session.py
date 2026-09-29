from __future__ import annotations
from dataclasses import dataclass
import random
import numpy as np
import torch
from .checkpoint import CheckpointManager
from .trainer_engine import TrainerEngine
from .training_loop import EarlyStopping
from .runs import RunConfig,run_id,TrainingRun,RunStore

def seed_everything(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends,"cudnn"):
        torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

@dataclass
class TrainingSummary:
    run_id:str
    best_epoch:int
    best_metric:float
    history:list[dict]
    checkpoint:str

def train_model(model,train_batches,val_batches,config:RunConfig,checkpoint_dir="models/checkpoints",device="cpu",patience=5):
    seed_everything(config.seed); model.to(device)
    if not train_batches: raise ValueError("training batches cannot be empty")
    if not val_batches: raise ValueError("validation batches cannot be empty")
    optimizer=torch.optim.AdamW(model.parameters(),lr=config.learning_rate)
    engine=TrainerEngine(model,optimizer,torch.nn.BCEWithLogitsLoss(),device)
    stopper=EarlyStopping(patience=patience,mode="max")
    ckpt=CheckpointManager(checkpoint_dir); history=[]; best_epoch=0; best_metric=float("-inf"); best_path=""
    run=TrainingRun(run_id(config),config.__dict__.copy()); run.start(); RunStore().append(run)
    try:
        for epoch in range(1,config.epochs+1):
            train=engine.train_epoch(train_batches); val=engine.validate(val_batches)
            history.append({"epoch":epoch,"train_loss":train.loss,"train_f1":train.metrics["f1"],
                "train_auroc":train.metrics["auroc"],"val_loss":val.loss,"val_f1":val.metrics["f1"],
                "val_accuracy":val.metrics["accuracy"],"val_auroc":val.metrics["auroc"],
                "val_pr_auc":val.metrics["pr_auc"],"val_brier":val.metrics["brier"],"val_ece":val.metrics["ece"]})
            if val.metrics["f1"]>best_metric:
                best_metric=val.metrics["f1"]; best_epoch=epoch
                best_path=ckpt.save(model,optimizer,epoch,best_metric,"best.pt")
            stopper.step(val.metrics["f1"])
            if stopper.stopped: break
        run.finish({"best_f1":best_metric,"best_epoch":best_epoch,"epochs_completed":len(history)})
    except Exception as exc:
        run.finish({"error_type":type(exc).__name__},status="failed")
        raise
    RunStore().append(run)
    return TrainingSummary(run_id(config),best_epoch,best_metric,history,best_path)
