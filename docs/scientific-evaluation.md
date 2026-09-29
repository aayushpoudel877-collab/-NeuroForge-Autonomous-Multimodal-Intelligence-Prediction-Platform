# Scientific Evaluation and Robustness

The evaluation layer separates predictive metrics from calibration and missing-modality robustness.

## Calibrated evaluation

The evaluate command supports a calibrate flag that fits a scalar temperature on supplied predictions and reports metrics after transformation. For a rigorous experiment, calibration should be fit on a validation set and measured on a separate test set; this CLI workflow is a development convenience.

## Robustness

The robustness command evaluates randomized modality availability at several dropout probabilities. Each sample retains at least one modality, and the model receives an explicit availability mask.

Example:

    neuroforge robustness --manifest data/synthetic/manifest.json --root data/synthetic

The report includes accuracy, F1, AUROC, PR-AUC, Brier and ECE at each dropout level.

## Reproducibility

Training now seeds Python, NumPy and PyTorch before model construction in the CLI and configures deterministic cuDNN behavior when available. Training rejects empty train/validation batches and records failed runs.

## Interpretation

AUROC and PR-AUC require both classes and are reported as 0.0 when undefined. Brier score and ECE are calibration-oriented measures where lower values are generally preferable, while AUROC and PR-AUC measure ranking quality. Metrics describe the supplied dataset only and are not real-world performance claims.
