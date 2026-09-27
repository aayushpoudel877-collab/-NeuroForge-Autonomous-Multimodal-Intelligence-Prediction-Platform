from __future__ import annotations
from .runs import RunConfig,TrainingRun,run_id,RunStore
from .selection import select_best

class ExperimentOrchestrator:
    def __init__(self,store=None): self.store=store or RunStore()
    def create_run(self,config:RunConfig):
        run=TrainingRun(run_id(config),config.__dict__.copy()); self.store.append(run); return run
    def record_result(self,run:TrainingRun,metrics:dict,success=True):
        run.finish(metrics,status="completed" if success else "failed"); self.store.append(run); return run
    def best(self,metric="f1"): return select_best(self.store.list(),metric)
