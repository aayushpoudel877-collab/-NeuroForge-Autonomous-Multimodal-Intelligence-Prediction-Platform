from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
from pathlib import Path

from .model_lifecycle import ArtifactStore, PromotionGate
from .registry import JsonModelRegistry, RegisteredModel


@dataclass(frozen=True)
class GovernanceDecision:
    model_name: str
    version: str
    approved: bool
    reason: str
    artifact_sha256: str
    metric: str
    value: float
    minimum: float
    created_at: str


class ModelGovernance:
    """Connect evaluated artifacts, promotion gates, registry state and audit history."""

    def __init__(
        self,
        artifact_store: ArtifactStore | None = None,
        registry: JsonModelRegistry | None = None,
        history_path: str = "models/governance.json",
        metric: str = "f1",
        minimum: float = 0.75,
    ):
        self.artifacts = artifact_store or ArtifactStore()
        self.registry = registry or JsonModelRegistry()
        self.history_path = Path(history_path)
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        self.gate = PromotionGate(metric=metric, minimum=minimum)

    def _history(self) -> list[dict]:
        return json.loads(self.history_path.read_text()) if self.history_path.exists() else []

    def _record(self, decision: GovernanceDecision) -> None:
        history = self._history()
        history.append(asdict(decision))
        self.history_path.write_text(json.dumps(history, indent=2))

    def evaluate(self, name: str, version: str, metrics: dict) -> GovernanceDecision:
        artifact = self.artifacts.get(name, version)
        gate = self.gate.evaluate(metrics)
        decision = GovernanceDecision(
            model_name=name,
            version=version,
            approved=bool(gate["approved"]),
            reason="promotion gate passed" if gate["approved"] else "promotion gate failed",
            artifact_sha256=artifact.sha256,
            metric=self.gate.metric,
            value=float(gate["value"]),
            minimum=float(gate["minimum"]),
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self._record(decision)
        return decision

    def register_and_evaluate(
        self, name: str, version: str, artifact_path: str, metrics: dict
    ) -> GovernanceDecision:
        artifact = self.artifacts.register_file(name, version, artifact_path, metrics)
        self.registry.register(
            RegisteredModel(
                name=name,
                version=version,
                artifact=artifact.path,
                metrics=metrics,
                stage="candidate",
            )
        )
        return self.evaluate(name, version, metrics)

    def promote_if_approved(self, decision: GovernanceDecision) -> None:
        if not decision.approved:
            raise ValueError(f"promotion denied for {decision.model_name}:{decision.version}")
        self.registry.promote(decision.model_name, decision.version, "production")

    def history(self) -> list[dict]:
        return self._history()
