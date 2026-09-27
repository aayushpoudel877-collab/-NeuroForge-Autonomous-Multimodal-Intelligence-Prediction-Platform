# Production Operations

## Dashboard

Open `/dashboard` to inspect service health, prediction counts, failure rate, latency, and recent alerts.

## Authentication

API-key enforcement is controlled by:
- `NEUROFORGE_REQUIRE_API_KEY=true`
- `NEUROFORGE_API_KEY_HASH=<sha256>`

Send the raw key as `X-API-Key`. Health, readiness, metrics, dashboard, and observability read endpoints remain available for platform monitoring.

Do not commit a real secret. Use a Kubernetes Secret or another secret manager.

## Kubernetes

The deployment includes:
- non-root execution and dropped Linux capabilities
- readiness and liveness probes
- rolling updates
- PodDisruptionBudget
- CPU-based HorizontalPodAutoscaler
- NetworkPolicy
- resource requests/limits

The example secret is a template only and must be replaced by a real secret-management workflow.

## CI quality gates

Pull requests and pushes run:
- unit/integration tests
- Ruff static checks
- Python bytecode compilation
- dependency vulnerability auditing with pip-audit
