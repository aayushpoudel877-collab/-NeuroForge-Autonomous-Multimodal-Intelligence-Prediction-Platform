import numpy as np
from neuroforge.deep_learning.metrics import binary_metrics

def test_binary_metrics():
    result=binary_metrics(np.array([8,-8,8,-8]),np.array([1,0,1,0]))
    assert result["accuracy"]==1.0
    assert result["f1"]==1.0
