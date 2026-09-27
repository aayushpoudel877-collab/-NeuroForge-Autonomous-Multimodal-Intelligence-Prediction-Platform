import hashlib
from fastapi.testclient import TestClient
from neuroforge.api import app

client=TestClient(app)

def test_dashboard():
    response=client.get("/dashboard")
    assert response.status_code==200
    assert "NeuroForge Operations" in response.text

def test_auth_disabled_by_default(monkeypatch):
    monkeypatch.delenv("NEUROFORGE_REQUIRE_API_KEY",raising=False)
    monkeypatch.delenv("NEUROFORGE_API_KEY_HASH",raising=False)
    response=client.post("/v1/predict",json={"text":"great result"})
    assert response.status_code==200

def test_auth_rejects_missing_key(monkeypatch):
    monkeypatch.setenv("NEUROFORGE_REQUIRE_API_KEY","true")
    monkeypatch.setenv("NEUROFORGE_API_KEY_HASH",hashlib.sha256(b"correct-key").hexdigest())
    response=client.post("/v1/predict",json={"text":"great result"})
    assert response.status_code==401

def test_auth_accepts_valid_key(monkeypatch):
    monkeypatch.setenv("NEUROFORGE_REQUIRE_API_KEY","true")
    monkeypatch.setenv("NEUROFORGE_API_KEY_HASH",hashlib.sha256(b"correct-key").hexdigest())
    response=client.post("/v1/predict",headers={"X-API-Key":"correct-key"},json={"text":"great result"})
    assert response.status_code==200
