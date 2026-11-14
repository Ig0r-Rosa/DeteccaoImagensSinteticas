#!/usr/bin/env bash
# Gera pares REAL→JSON→IA com retomada (Ctrl+C e rode de novo).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PY="${ROOT}/.venv/bin/python"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -x "$PY" ]]; then
  echo "Crie o .venv antes."
  exit 1
fi

echo "Projeto: $ROOT"
echo "Args: $*"
exec "$PY" -m cod.pares.pipeline "$@"
