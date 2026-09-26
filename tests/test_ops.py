from neuroforge.monitoring import PredictionMonitor
from neuroforge.model_registry import InMemoryModelRegistry,ModelArtifact

def test_monitor_snapshot():
    m=PredictionMonitor();m.record(.8,10);m.record(.2,20)
    s=m.snapshot();assert s["samples"]==2;assert s["mean_latency_ms"]==15

def test_registry_promotion():
    r=InMemoryModelRegistry();r.register(ModelArtifact("fusion","1.0",metric=.91));r.promote("fusion","1.0")
    assert r.get("fusion","1.0").stage=="production"
