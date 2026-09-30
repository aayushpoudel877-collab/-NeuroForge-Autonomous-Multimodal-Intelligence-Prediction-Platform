# Model Governance & Safe Promotion

NeuroForge now has a governance layer between research output and production registry state.

## Lifecycle

1. A trained checkpoint is registered as an immutable artifact copy.
2. Its SHA-256 digest is recorded.
3. The artifact is registered as a candidate model.
4. A configurable metric gate evaluates the supplied evaluation metrics.
5. The governance decision is written to an audit history.
6. Only an approved candidate can be promoted to production.

The gate is explicit and configurable. The default implementation uses F1 with a minimum of 0.75, but this threshold is not evidence that the model is suitable for any particular real-world application.

## Safety properties

- Production promotion cannot happen through this workflow unless the gate passes.
- Artifact content is copied into the artifact store and hashed.
- Governance decisions retain the metric value, threshold, artifact digest and timestamp.
- Rejected candidates remain auditable.

This is a development governance baseline. Production deployments should additionally use signed artifacts, external durable storage, approval workflows and organization-specific validation requirements.
