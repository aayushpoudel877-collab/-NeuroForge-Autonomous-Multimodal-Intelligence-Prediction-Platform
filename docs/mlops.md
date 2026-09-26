# MLOps Architecture

NeuroForge separates model development from serving through four operational layers:

1. **Experiment tracking** records run IDs, parameters, metrics, and timestamps.
2. **Dataset lineage** fingerprints schemas and records dataset versions and row counts.
3. **Model registry** stores model versions, artifacts, metrics, and deployment stages.
4. **Runtime monitoring** tracks prediction distributions, latency, errors, and drift indicators.

Probability calibration and feature-drift utilities are included as lightweight reference implementations. Production deployments should persist telemetry in a managed metrics store, use signed model artifacts, and enforce organization-specific retention and access policies.
