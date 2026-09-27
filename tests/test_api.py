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

def test_prometheus_metrics():
    response=client.get("/metrics/prometheus")
    assert response.status_code==200
    assert "neuroforge_predictions_total" in response.text

def test_batch_prediction():
    response=client.post("/v1/batch-predict",json={"requests":[{"text":"great result"},{"text":"bad failure"}]})
    assert response.status_code==200
    assert len(response.json())==2

def test_request_id_and_trace():
    response=client.post("/v1/predict",headers={"X-Request-ID":"test-request-123"},json={"text":"great result","metadata":{"model_version":"demo-1","training_run":"run-1"}})
    assert response.status_code==200
    assert response.headers["X-Request-ID"]=="test-request-123"
    traces=client.get("/observability/traces").json()
    assert any(x["request_id"]=="test-request-123" for x in traces)

def test_models_endpoint():
    response=client.get("/v1/models")
    assert response.status_code==200
    assert isinstance(response.json(),list)
