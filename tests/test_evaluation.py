import numpy as np
import pytest
torch=pytest.importorskip("torch")
from neuroforge.deep_learning.unified_model import NeuroForgeMultimodalModel
from neuroforge.deep_learning.evaluation import evaluate_model,evaluate_calibrated

def _batch():
    return {"text":np.zeros((2,128),dtype=np.int64),"image":np.zeros((2,3,64,64),dtype=np.float32),
        "audio":np.zeros((2,1,16000),dtype=np.float32),"series":np.zeros((2,64,1),dtype=np.float32),
        "labels":np.array([0,1],dtype=np.float32)}

def test_multimodal_evaluation():
    report=evaluate_model(NeuroForgeMultimodalModel(),[_batch()])
    assert report["samples"]==2 and 0<=report["f1"]<=1
    assert "auroc" in report and "ece" in report

def test_calibrated_evaluation():
    report=evaluate_calibrated(NeuroForgeMultimodalModel(),[_batch()])
    assert report["samples"]==2 and .05<=report["temperature"]<=20
