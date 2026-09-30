from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch

from .evaluation import collect_predictions
from .metrics import binary_metrics
from .runs import RunConfig
from .training_session import TrainingSummary, train_model


@dataclass(frozen=True)
class ExperimentSpec:
    learning_rates: tuple[float, ...] = (1e-3,)
    batch_sizes: tuple[int, ...] = (4,)
    epochs: int = 3
    seeds: tuple[int, ...] = (42, 43, 44)
    model_name: str = "neuroforge_multimodal"
    bootstrap_samples: int = 1000

    def __post_init__(self):
        if not self.learning_rates or any(x <= 0 for x in self.learning_rates):
            raise ValueError("learning_rates must contain positive values")
        if not self.batch_sizes or any(x <= 0 for x in self.batch_sizes):
            raise ValueError("batch_sizes must contain positive values")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if not self.seeds:
            raise ValueError("at least one seed is required")
        if self.bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be at least 100")

    def configs(self) -> list[RunConfig]:
        return [
            RunConfig(
                model_name=self.model_name,
                learning_rate=lr,
                batch_size=batch_size,
                epochs=self.epochs,
                seed=seed,
            )
            for lr in self.learning_rates
            for batch_size in self.batch_sizes
            for seed in self.seeds
        ]


def bootstrap_ci(
    values: Iterable[float],
    confidence: float = 0.95,
    samples: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    values = np.asarray(list(values), dtype=float)
    if values.size == 0:
        raise ValueError("values cannot be empty")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if samples < 100:
        raise ValueError("samples must be at least 100")
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=float)
    for i in range(samples):
        means[i] = rng.choice(values, size=values.size, replace=True).mean()
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "lower": float(np.quantile(means, alpha)),
        "upper": float(np.quantile(means, 1.0 - alpha)),
        "n": int(values.size),
    }


def aggregate_seed_metrics(
    results: list[dict],
    metrics: tuple[str, ...] = ("f1", "auroc", "pr_auc", "brier", "ece"),
    bootstrap_samples: int = 1000,
) -> dict:
    aggregate = {}
    for metric in metrics:
        values = [float(r[metric]) for r in results if metric in r]
        if values:
            aggregate[metric] = bootstrap_ci(
                values, samples=bootstrap_samples, seed=42 + len(metric)
            )
    return aggregate


def select_experiment(results: list[dict], metric: str = "f1", maximize: bool = True) -> dict | None:
    valid = [r for r in results if r.get("status") == "completed" and metric in r]
    if not valid:
        return None
    return sorted(valid, key=lambda r: float(r[metric]), reverse=maximize)[0]


def evaluate_from_model(model, batches, device="cpu", ablation=None) -> dict:
    logits, labels = collect_predictions(model, batches, device=device, ablation=ablation)
    return {"samples": int(len(labels)), **binary_metrics(logits, labels)}


def modality_ablation_matrix(model, batches, device="cpu") -> dict[str, dict]:
    modes = ("full", "text", "image", "audio", "temporal")
    return {
        mode: evaluate_from_model(
            model, batches, device=device, ablation=None if mode == "full" else mode
        )
        for mode in modes
    }


class ResearchExperimentEngine:
    """Runs a reproducible hyperparameter and seed grid on held-out test data."""

    def __init__(
        self,
        model_factory: Callable[[], torch.nn.Module],
        train_batches_factory: Callable[[RunConfig], list[dict]],
        val_batches_factory: Callable[[RunConfig], list[dict]],
        test_batches: list[dict],
        checkpoint_dir: str = "models/checkpoints/research",
        device: str = "cpu",
    ):
        self.model_factory = model_factory
        self.train_batches_factory = train_batches_factory
        self.val_batches_factory = val_batches_factory
        self.test_batches = test_batches
        self.checkpoint_dir = checkpoint_dir
        self.device = device

    def run(self, spec: ExperimentSpec) -> dict:
        results = []
        for config in spec.configs():
            try:
                from .training_session import seed_everything
                from .calibration_fit import fit_temperature

                seed_everything(config.seed)
                model = self.model_factory()
                train_batches = self.train_batches_factory(config)
                val_batches = self.val_batches_factory(config)
                summary: TrainingSummary = train_model(
                    model,
                    train_batches,
                    val_batches,
                    config,
                    checkpoint_dir=self.checkpoint_dir,
                    device=self.device,
                )
                state = torch.load(
                    summary.checkpoint, map_location=self.device, weights_only=False
                )
                model.load_state_dict(state["model"])
                model.to(self.device)

                test_logits, test_labels = collect_predictions(
                    model, self.test_batches, device=self.device
                )
                test = {
                    "samples": int(len(test_labels)),
                    **binary_metrics(test_logits, test_labels),
                }

                val_logits, val_labels = collect_predictions(
                    model, val_batches, device=self.device
                )
                scaler = fit_temperature(val_logits, val_labels)
                probabilities = scaler.transform_logits(test_logits)
                probabilities = np.clip(probabilities, 1e-6, 1 - 1e-6)
                calibrated_logits = np.log(probabilities / (1 - probabilities))
                calibrated = binary_metrics(calibrated_logits, test_labels)

                results.append(
                    {
                        "status": "completed",
                        "run_id": summary.run_id,
                        "learning_rate": config.learning_rate,
                        "batch_size": config.batch_size,
                        "epochs": config.epochs,
                        "seed": config.seed,
                        "checkpoint": summary.checkpoint,
                        "validation_f1": float(summary.best_metric),
                        **test,
                        "calibrated": calibrated,
                        "temperature": float(scaler.temperature),
                    }
                )
            except Exception as exc:
                results.append(
                    {
                        "status": "failed",
                        "learning_rate": config.learning_rate,
                        "batch_size": config.batch_size,
                        "epochs": config.epochs,
                        "seed": config.seed,
                        "error_type": type(exc).__name__,
                    }
                )

        completed = [r for r in results if r["status"] == "completed"]
        grouped = {}
        for result in completed:
            key = (result["learning_rate"], result["batch_size"], result["epochs"])
            grouped.setdefault(key, []).append(result)

        comparisons = []
        for key, group in grouped.items():
            comparisons.append(
                {
                    "learning_rate": key[0],
                    "batch_size": key[1],
                    "epochs": key[2],
                    "seeds": [r["seed"] for r in group],
                    "validation": aggregate_seed_metrics(
                        group, metrics=("validation_f1",), bootstrap_samples=spec.bootstrap_samples
                    ),
                    "test_metrics": aggregate_seed_metrics(
                        group, bootstrap_samples=spec.bootstrap_samples
                    ),
                }
            )
        comparisons.sort(
            key=lambda x: x["validation"].get("validation_f1", {}).get("mean", -math.inf),
            reverse=True,
        )
        best_run = max(
            completed,
            key=lambda x: float(x.get("validation_f1", -math.inf)),
            default=None,
        )
        return {
            "experiment": asdict(spec),
            "results": results,
            "comparison": comparisons,
            "best": comparisons[0] if comparisons else None,
            "best_run": best_run,
        }


def save_research_report(report: dict, path: str = "reports/research-experiment.json") -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2))
    return str(p)
