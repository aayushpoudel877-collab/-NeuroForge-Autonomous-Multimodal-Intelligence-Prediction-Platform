# Development

python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
uvicorn neuroforge.api:app --reload

The shipped adapters are deterministic and lightweight so CI works without downloading large foundation models. They are explicit extension points for production deep-learning models.
