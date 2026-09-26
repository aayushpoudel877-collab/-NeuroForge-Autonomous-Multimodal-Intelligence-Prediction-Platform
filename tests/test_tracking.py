from neuroforge.experiments import ExperimentTracker
from neuroforge.data.lineage import schema_fingerprint

def test_schema_fingerprint_stable():
    assert schema_fingerprint({"a":"float"})==schema_fingerprint({"a":"float"})

def test_experiment_log(tmp_path):
    t=ExperimentTracker(str(tmp_path/"runs.jsonl"));t.log("r1","fusion",{"lr":.01},{"f1":.8})
    assert t.runs()[0]["run_id"]=="r1"
