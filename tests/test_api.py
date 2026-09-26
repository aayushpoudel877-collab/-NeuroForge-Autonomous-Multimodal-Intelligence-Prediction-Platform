from fastapi.testclient import TestClient
from neuroforge.api import app

client=TestClient(app)

def test_health_and_ready():
    assert client.get("/health").status_code==200
    body=client.get("/ready").json()
    assert body["ready"] is True

def test_metrics_endpoint():
    body=client.get("/metrics").json()
    assert "api" in body and "inference" in body and "health" in body

def test_batch_prediction():
    response=client.post("/v1/batch-predict",json={"requests":[{"text":"great result"},{"text":"bad failure"}]})
    assert response.status_code==200
    assert len(response.json())==2

def test_models_endpoint():
    response=client.get("/v1/models")
    assert response.status_code==200
    assert isinstance(response.json(),list)
