# Research Experiments & Model Selection

NeuroForge now provides a reusable research experiment engine for grid search across learning rates, batch sizes and random seeds.

## Scientific protocol

Each configuration is evaluated as:

1. Construct a fresh model after seeding.
2. Train only on the training split.
3. Select the best checkpoint using validation F1.
4. Evaluate that checkpoint once on the held-out test split.
5. Fit temperature scaling on validation predictions only.
6. Apply the fitted temperature to untouched test predictions.
7. Aggregate repeated seeds with bootstrap confidence intervals.

This keeps test data out of checkpoint selection and calibration fitting.

## ExperimentSpec

ExperimentSpec describes the search grid:

- learning_rates
- batch_sizes
- epochs
- seeds
- bootstrap_samples

ResearchExperimentEngine receives model/data factories so experiments remain independent of one dataset implementation.

## Interpreting results

The engine reports F1, AUROC, PR-AUC, Brier score and ECE. For multiple seeds, it reports mean, standard deviation and a bootstrap confidence interval for the mean.

The best configuration is selected by mean F1 across its completed seeds. This is a descriptive selection rule for the experiment, not evidence of real-world superiority.

Synthetic data remains a development fixture. Results on it must not be presented as production performance.

## Failure handling

A failed configuration is retained in the report with its error type and does not contribute to aggregate metrics.

## Reproducibility

The engine seeds before model construction and passes each seed into RunConfig. Keep data splitting deterministic and avoid changing preprocessing between configurations when comparing models.
