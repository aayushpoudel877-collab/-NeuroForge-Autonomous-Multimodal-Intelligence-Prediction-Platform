import numpy as np
from neuroforge.calibration import TemperatureScaler
from neuroforge.drift import DriftDetector
from neuroforge.security import hash_api_key,verify_api_key

def test_calibration_bounds():
    p=TemperatureScaler(2).transform([.1,.5,.9]);assert np.all((p>0)&(p<1))

def test_drift_detector():
    d=DriftDetector();assert d.score(np.zeros(100),np.ones(100))>0

def test_key_verification():
    h=hash_api_key("secret");assert verify_api_key("secret",h);assert not verify_api_key("wrong",h)
