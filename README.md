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
