from neuroforge.deep_learning.orchestrator import ExperimentOrchestrator
from neuroforge.deep_learning.runs import RunConfig,RunStore

def test_orchestrator_records_and_selects(tmp_path):
    store=RunStore(tmp_path/"runs.jsonl")
    o=ExperimentOrchestrator(store)
    run=o.create_run(RunConfig())
    o.record_result(run,{"f1":.81})
    assert o.best()["metrics"]["f1"]==.81
