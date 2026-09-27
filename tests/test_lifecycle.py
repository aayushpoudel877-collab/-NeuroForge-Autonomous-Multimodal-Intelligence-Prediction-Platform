from neuroforge.model_lifecycle import PromotionGate,ArtifactStore,LifecycleHistory

def test_promotion_gate():
    assert PromotionGate(minimum=.8).evaluate({"f1":.81})["approved"]
    assert not PromotionGate(minimum=.8).evaluate({"f1":.79})["approved"]

def test_artifact_store(tmp_path):
    source=tmp_path/"model.bin";source.write_bytes(b"neuroforge")
    artifact=ArtifactStore(tmp_path/"artifacts").register_file("demo","1",source,{"f1":.9})
    assert len(artifact.sha256)==64

def test_history(tmp_path):
    history=LifecycleHistory(tmp_path/"history.json")
    history.record({"event":"promote","version":"1"})
    assert history.list()[0]["event"]=="promote"
