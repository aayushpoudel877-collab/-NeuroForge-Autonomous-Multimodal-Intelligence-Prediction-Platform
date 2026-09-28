import numpy as np
import pytest
torch=pytest.importorskip("torch")
from neuroforge.deep_learning.calibration_fit import fit_temperature
from neuroforge.deep_learning.runs import RunConfig,run_id

def test_temperature_fit_is_valid():
    scaler=fit_temperature(np.array([-2.,2.,-1.,1.]),np.array([0.,1.,0.,1.]))
    assert .05<=scaler.temperature<=20

def test_run_id_is_deterministic():
    config=RunConfig(seed=7,epochs=2)
    assert run_id(config)==run_id(config)
