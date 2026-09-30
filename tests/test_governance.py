from neuroforge.governance import ModelGovernance
from neuroforge.model_lifecycle import ArtifactStore
from neuroforge.registry import JsonModelRegistry


def test_governance_approves_and_promotes(tmp_path):
    source = tmp_path / "model.pt"
    source.write_bytes(b"weights")
    governance = ModelGovernance(
        artifact_store=ArtifactStore(tmp_path / "artifacts"),
        registry=JsonModelRegistry(tmp_path / "registry.json"),
        history_path=str(tmp_path / "governance.json"),
        minimum=.8,
    )
    decision = governance.register_and_evaluate(
        "demo", "1", str(source), {"f1": .9}
    )
    assert decision.approved
    governance.promote_if_approved(decision)
    assert governance.registry.list()[0]["stage"] == "production"


def test_governance_rejects_below_gate(tmp_path):
    source = tmp_path / "model.pt"
    source.write_bytes(b"weights")
    governance = ModelGovernance(
        artifact_store=ArtifactStore(tmp_path / "artifacts"),
        registry=JsonModelRegistry(tmp_path / "registry.json"),
        history_path=str(tmp_path / "governance.json"),
        minimum=.8,
    )
    decision = governance.register_and_evaluate(
        "demo", "1", str(source), {"f1": .79}
    )
    assert not decision.approved
