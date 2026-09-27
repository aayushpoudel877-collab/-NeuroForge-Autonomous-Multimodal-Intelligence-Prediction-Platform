from fastapi.testclient import TestClient
from neuroforge.api import app

client=TestClient(app)

def test_dashboard():
    response=client.get("/dashboard")
    assert response.status_code==200
    assert "NeuroForge Operations" in response.text

def test_auth_disabled_by_default(monkeypatch):
    monkeypatch.delenv("NEUROFORGE_REQUIRE_API_KEY",raising=False)
    response=client.post("/v1/predict",json={"text":"great result"})
    assert response.status_code==200
