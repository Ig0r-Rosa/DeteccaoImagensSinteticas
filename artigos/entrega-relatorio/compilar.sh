#!/usr/bin/env bash
# Compila main.tex → main.pdf (configs/ + suporte/ + anexos/)
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

export TEXINPUTS="${DIR}/configs//:${TEXINPUTS:-}"
export BSTINPUTS="${DIR}/configs//:${BSTINPUTS:-}"
export BIBINPUTS="${DIR}/configs//:${BIBINPUTS:-}"

TEX="main"
PDF="${TEX}.pdf"
LOG="${TEX}.build.log"

if ! command -v pdflatex >/dev/null 2>&1; then
  echo "Erro: pdflatex não encontrado."
  echo "Instale: sudo apt install texlive-latex-base texlive-latex-recommended texlive-fonts-recommended texlive-bibtex-extra texlive-lang-portuguese texlive-latex-extra texlive-extra-utils"
  exit 1
fi

run_pdflatex() {
  if ! pdflatex -interaction=nonstopmode -halt-on-error "${TEX}.tex" >"${LOG}" 2>&1; then
    echo "Falhou no pdflatex. Últimas linhas:"
    tail -50 "${LOG}"
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

# Remove auxiliares
rm -f "${TEX}.aux" "${TEX}.bbl" "${TEX}.blg" "${TEX}.log" "${TEX}.out" "${TEX}.toc" "${TEX}.build.log"

PAGES=$(pdfinfo "${PDF}" 2>/dev/null | awk '/Pages/ {print $2}' || echo "?")
echo "OK: ${DIR}/${PDF} (${PAGES} páginas)"
ls -lh "${PDF}"
