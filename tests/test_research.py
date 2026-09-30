import pytest

from neuroforge.deep_learning.research import (
    ExperimentSpec,
    aggregate_seed_metrics,
    bootstrap_ci,
    select_experiment,
)


def test_bootstrap_ci_is_deterministic():
    values = [0.7, 0.8, 0.9]
    first = bootstrap_ci(values, samples=200)
    second = bootstrap_ci(values, samples=200)
    assert first == second
    assert first["n"] == 3
    assert first["lower"] <= first["mean"] <= first["upper"]


def test_experiment_grid_and_selection():
    spec = ExperimentSpec(
        learning_rates=(1e-3, 5e-4),
        batch_sizes=(4, 8),
        epochs=2,
        seeds=(42, 43),
        bootstrap_samples=200,
    )
    assert len(spec.configs()) == 8
    results = [
        {"status": "completed", "seed": 42, "f1": 0.95, "validation_f1": 0.6, "auroc": 0.7},
        {"status": "completed", "seed": 43, "f1": 0.7, "validation_f1": 0.8, "auroc": 0.9},
        {"status": "failed", "seed": 44, "f1": 1.0},
    ]
    assert select_experiment(results)["f1"] == pytest.approx(0.7)
    aggregate = aggregate_seed_metrics(results[:2], bootstrap_samples=200)
    assert aggregate["f1"]["n"] == 2
