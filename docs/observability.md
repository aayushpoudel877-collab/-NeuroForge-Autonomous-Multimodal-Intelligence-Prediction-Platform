# Observability and Reliability

NeuroForge now records structured inference traces and threshold-based alerts without requiring an external observability service.

## Inference traces

Each trace can include:
- request ID and timestamps
- status and latency
- modalities used
- prediction and probability
- model name/version
- dataset snapshot and training run identifiers
- artifact SHA-256
- error type

The default JSONL sink is `logs/inference.jsonl`.

## Alerts

The alert layer provides dependency-light threshold evaluation and a JSONL sink at `logs/alerts.jsonl`. It is an event layer that can later connect to external notification systems.

## API endpoints

- `/metrics` returns JSON operational metrics.
- `/metrics/prometheus` returns Prometheus-compatible text exposition.
- `/observability/traces` returns recent structured traces.
- `/observability/alerts` returns recent alerts.

## Lineage

Prediction metadata may carry `model_name`, `model_version`, `dataset_snapshot`, `training_run`, and `artifact_sha256`. These identifiers are stored with the inference trace so model, data, training, and artifact provenance can be correlated.

## Request correlation

Clients may send `X-Request-ID`. If absent, NeuroForge generates a UUID-based request ID and returns it in the response header.

## Reliability

Telemetry writes are isolated from inference. A logging failure does not fail a successful prediction request.
