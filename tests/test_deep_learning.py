import pytest

torch=pytest.importorskip("torch")
from neuroforge.deep_learning.trainer import SyntheticMultimodalTrainer

def test_neural_training_metrics():
    _,metrics=SyntheticMultimodalTrainer().fit(epochs=2)
    assert set(metrics)=={"accuracy","f1","roc_auc"}
    assert all(0.0<=v<=1.0 for v in metrics.values())
