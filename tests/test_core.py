from fastapi.testclient import TestClient
from neuroforge.api import app
from neuroforge.fusion import ConfidenceWeightedFusion
def test_fusion():
    o=ConfidenceWeightedFusion().fuse([("a",.9,.9),("b",.1,.1)])
    assert o.probability>.7 and o.weights["a"]>o.weights["b"]
def test_health():
    r=TestClient(app).get("/health");assert r.status_code==200
def test_prediction():
    r=TestClient(app).post("/v1/predict",json={"text":"growth is rising","series":[1,2,3,4,5]})
    assert r.status_code==200 and 0<=r.json()["probability"]<=1
