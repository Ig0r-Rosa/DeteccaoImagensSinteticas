#!/usr/bin/env bash
# Sempre usa o .venv (Python 3.11 + ThunderSVM). Não use `streamlit` do sistema.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
exec "$ROOT/.venv/bin/streamlit" run app.py "$@"
