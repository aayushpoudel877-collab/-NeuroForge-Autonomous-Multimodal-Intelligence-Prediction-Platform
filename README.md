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
