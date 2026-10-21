#!/usr/bin/env bash
# Recompila o PDF da entrega 02 (entrega02.tex + referencias-e2.bib)
# Não altera main.tex / Entrega01.pdf

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

TEX="entrega02"
PDF="${TEX}.pdf"
LOG="${TEX}.build.log"

if ! command -v pdflatex >/dev/null 2>&1; then
  echo "Erro: pdflatex não encontrado."
  exit 1
fi

run_pdflatex() {
  if ! pdflatex -interaction=nonstopmode -halt-on-error "${TEX}.tex" >"${LOG}" 2>&1; then
    echo "Falhou no pdflatex. Últimas linhas:"
    tail -40 "${LOG}"
    exit 1
  fi
}

echo "==> 1/4 pdflatex (passagem 1)"
run_pdflatex

echo "==> 2/4 bibtex"
bibtex "${TEX}" >>"${LOG}" 2>&1 || true

echo "==> 3/4 pdflatex (passagem 2)"
run_pdflatex

echo "==> 4/4 pdflatex (passagem 3)"
run_pdflatex

echo "OK: ${DIR}/${PDF}"
ls -lh "${PDF}"
