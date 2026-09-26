# Security Baseline

The repository includes API-key hashing/verification primitives and a production configuration profile.

Recommended deployment controls include TLS termination, secret injection through the deployment platform, least-privilege service accounts, request-size limits, rate limiting, structured audit logs, dependency scanning, and network isolation for model infrastructure.

Secrets must never be committed to the repository. The included security utilities are building blocks, not a replacement for an identity provider or production secrets manager.
