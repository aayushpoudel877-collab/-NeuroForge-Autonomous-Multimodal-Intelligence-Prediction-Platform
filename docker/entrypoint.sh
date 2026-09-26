#!/bin/sh
set -eu
exec uvicorn neuroforge.api:app --host 0.0.0.0 --port 8000
