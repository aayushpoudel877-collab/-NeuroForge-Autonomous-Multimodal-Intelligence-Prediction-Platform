# NeuroForge — Autonomous Multimodal Intelligence & Prediction Platform

NeuroForge is an end-to-end ML/DL platform that turns text, images, audio and time-series signals into calibrated predictions, confidence estimates and machine-generated explanations.

## Core capabilities

- Multimodal ingestion and validation
- Native Transformer/CNN temporal and cross-modal neural stack
- Optional pretrained Transformer, vision and Wav2Vec2 backbones
- Learned missing-modality representations
- Accuracy, F1, AUROC, PR-AUC, Brier and ECE evaluation
- Temperature calibration and model-card generation
- FastAPI inference service
- Reproducible training, checkpoints and experiment tracking
- Docker, Kubernetes and CI/CD production foundations

The repository intentionally separates lightweight serving adapters from the trainable PyTorch stack. Synthetic data is provided only as a development fixture; no real-world performance claim is made.

## Run locally

Create a virtual environment and install the development stack, then run the API with:
uvicorn neuroforge.api:app --reload

For neural training, install the ml extra. For optional pretrained backbones, install the advanced extra.

## Neural workflow

neuroforge prepare-demo --output data/synthetic --samples 24
neuroforge train --manifest data/synthetic/manifest.json --root data/synthetic --epochs 3
neuroforge evaluate --manifest data/synthetic/manifest.json --root data/synthetic --checkpoint models/checkpoints/best.pt
neuroforge benchmark --manifest data/synthetic/manifest.json --root data/synthetic --checkpoint models/checkpoints/best.pt

The training path includes deterministic seeding, train/validation metrics, early stopping, best-checkpoint selection and persistent run records.

## Advanced Model Intelligence

Optional pretrained adapters are available for Hugging Face text Transformers, torchvision vision models and torchaudio Wav2Vec2. They load weights only when explicitly instantiated. The unified multimodal model also supports a four-value modality mask and learned embeddings for missing inputs.

Evaluation now reports threshold metrics plus AUROC, PR-AUC, Brier score and expected calibration error. Model cards can be generated with neuroforge.model_card.

See docs/advanced-intelligence.md for the advanced layer and its scientific limitations.


## Scientific Evaluation & Robustness

The latest training layer adds:
- reproducible initialization before model construction
- deterministic cuDNN settings when available
- explicit rejection of empty training/validation batches
- failed-run recording in the experiment store
- calibrated evaluation through temperature scaling
- randomized missing-modality robustness testing
- modality dropout reports at multiple probabilities
- expanded training history with AUROC, PR-AUC, Brier and ECE

Development examples:

    neuroforge evaluate --manifest data/synthetic/manifest.json --root data/synthetic --calibrate
    neuroforge robustness --manifest data/synthetic/manifest.json --root data/synthetic

For rigorous scientific evaluation, calibration should use a validation split and final metrics should be measured on an untouched test split. The synthetic dataset remains a development fixture.


## Research Experiments & Model Selection

The research layer automates multi-seed hyperparameter experiments while keeping the held-out test set out of checkpoint selection and calibration fitting.

Example:

    neuroforge experiment --manifest data/synthetic/manifest.json --root data/synthetic --epochs 3 --learning-rates 0.001,0.0005 --batch-sizes 4,8 --seeds 42,43,44

Each configuration:
- builds a fresh model after deterministic seeding
- trains on the training split
- selects the checkpoint using validation F1
- evaluates once on the untouched test split
- fits temperature scaling on validation predictions only
- reports calibrated test metrics
- aggregates repeated seeds with bootstrap confidence intervals
- records failed configurations rather than silently dropping them

The experiment report also includes a modality-ablation matrix for the selected run and an automatic research model card. Synthetic data remains a development fixture and must not be interpreted as production evidence.


## Model Governance & Safe Promotion

NeuroForge v0.5.0 adds a governed path from research artifacts to production registry state:

**checkpoint → immutable artifact copy → SHA-256 digest → candidate registry entry → metric gate → audited decision → production promotion**

Use `neuroforge.governance.ModelGovernance` to register and evaluate a checkpoint. Promotion is permitted only when the configured gate passes. Governance history records the model version, artifact digest, metric value, threshold and timestamp.

This governance layer is a development baseline. Production environments should add signed artifacts, durable external storage, human approval workflows and application-specific validation requirements.
