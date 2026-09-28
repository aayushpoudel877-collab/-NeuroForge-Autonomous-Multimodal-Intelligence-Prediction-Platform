# NeuroForge — Autonomous Multimodal Intelligence & Prediction Platform

NeuroForge is an end-to-end ML/DL platform that turns **text, images, audio and time-series signals** into calibrated predictions, confidence estimates and machine-generated explanations.

## Core capabilities

- Multimodal ingestion and validation
- Text, image, audio and time-series model adapters
- Confidence-weighted multimodal fusion
- Uncertainty and anomaly scoring
- Explainable prediction responses
- FastAPI inference service
- Reproducible training/evaluation utilities
- Docker deployment and GitHub Actions CI
- Clear extension points for transformers, CNNs/ViTs, speech models and temporal deep learning

## Architecture

```
Text ───────┐
Image ──────┤
Audio ──────┼──> Modality Encoders ─> Confidence Fusion
Series ─────┘                              │
                                          ▼
                              Prediction + Uncertainty
                                          │
                                          ▼
                                   Explanation Layer
```

The repository uses lightweight deterministic adapters so the project can run in CI without downloading large foundation models. Each adapter follows a stable interface that can be replaced by a production deep-learning model.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
uvicorn neuroforge.api:app --reload
```

Then open `http://127.0.0.1:8000/docs`.

### Example

```bash
curl -X POST http://127.0.0.1:8000/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"growth is rising and stable","series":[1,2,3,4,5]}'
```

## Test

```bash
pytest -q
ruff check .
```

## Project structure

- `neuroforge/models/` — modality-specific encoders
- `neuroforge/fusion.py` — multimodal fusion
- `neuroforge/pipeline.py` — orchestration
- `neuroforge/explainability.py` — evidence and uncertainty
- `neuroforge/training.py` — reproducible ML training demo
- `neuroforge/api.py` — serving layer
- `configs/` — runtime configuration
- `docs/` — architecture and development notes
- `tests/` — automated verification

## Production roadmap

1. Transformer/ViT/wav2vec2 model adapters
2. Cross-modal attention and learned fusion
3. Model registry and experiment tracking
4. Drift monitoring and automated retraining
5. Feature store/vector retrieval
6. Distributed inference and observability


## Latest Engineering Expansion

The platform now also contains a separated trainable PyTorch layer with modality MLP encoders, learned confidence-style fusion gates, reproducible synthetic training, and Accuracy/F1/ROC-AUC evaluation. MLOps foundations were added through an inference monitor and model-registry abstraction with tests. These components are intentionally decoupled from the serving adapters so they can mature into real Transformer, vision, speech, and temporal models without redesigning the API.

See docs/deep-learning.md and configs/neural.yaml for the neural architecture and training configuration.


### Production Engineering Layer

Recent additions include persistent model registration, experiment tracking, dataset schema lineage, probability calibration, feature drift detection, runtime health state, API operational models, security primitives, production configuration, Kubernetes deployment manifests, and a hardened container entrypoint. These are intentionally lightweight reference components that can be connected to managed infrastructure in a real deployment.


## Observability and reliability

The current production layer includes:
- structured JSONL inference traces with request correlation IDs
- model, dataset, training-run, and artifact lineage fields
- dependency-light threshold alerts
- JSON and Prometheus-style metrics
- recent trace and alert API endpoints
- persisted artifact metadata with SHA-256 integrity records
- safer generic 500 responses that avoid exposing internal exception text
- multimodal dataset-to-trainer batch validation


## Production hardening

- Operations dashboard at `/dashboard`
- Optional API-key enforcement through environment variables
- Prometheus-style metrics endpoint
- CI tests, Ruff checks, bytecode compilation, and dependency auditing
- Non-root Docker execution
- Kubernetes readiness/liveness probes
- Rolling deployment strategy, PDB, HPA, and NetworkPolicy
- Runtime telemetry files excluded from version control
- Kubernetes secret template for API-key configuration

The repository remains a reference architecture: the demo serving adapters are lightweight, while the trainable multimodal PyTorch stack is separated for real dataset/model integration.


## Real ML/DL Integration Layer

The platform now includes a concrete data-to-model path:
- modality preprocessing for text, images, audio, and temporal series
- manifest-backed multimodal dataset adapter
- deterministic synthetic dataset generator for local development
- tensor batch collation matching the neural model contract
- evaluation runner with accuracy, precision, recall, and F1
- modality-ablation benchmark for diagnostic comparisons
- `neuroforge` CLI commands for demo-data preparation, evaluation, and benchmarking
- automated tests covering preprocessing, collation, and evaluation

Example:
```bash
neuroforge prepare-demo --output data/synthetic --samples 24
neuroforge evaluate --manifest data/synthetic/manifest.json --root data/synthetic
neuroforge benchmark --manifest data/synthetic/manifest.json --root data/synthetic
```

The generated dataset is intentionally synthetic. Real-world datasets and pretrained backbones are not claimed to be bundled; production integrations should provide licensed data, trained checkpoints, and dataset-specific preprocessing.


## Training & Experiment Execution

The project now has a complete local training path from manifest data to a selected checkpoint:
- deterministic seeding
- train/validation epoch execution
- validation metrics every epoch
- early stopping
- best-checkpoint selection by validation F1
- deterministic experiment run IDs
- temperature-scaling calibration utility
- `neuroforge train` CLI workflow

Example:
```bash
neuroforge train --manifest data/synthetic/manifest.json --root data/synthetic --epochs 3
```

This provides the engineering path for real datasets while keeping scientific performance claims separate from the synthetic development fixture.
