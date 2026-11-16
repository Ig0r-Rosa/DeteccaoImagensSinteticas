# Entrega — Relatório (Moodle)

Relatório final (meta ~15–17 páginas; faixa Moodle 10–20) + anexo do poster.

- Capa (pág. 1) só com título/resumos; sumário na pág. 2.
- Protocolo único de treino/teste: **4.000 / 1.000** pares.

## Estrutura

```
entrega-relatorio/
├── main.tex          ← fonte
├── main.pdf          ← gerar com ./compilar.sh
├── compilar.sh
├── configs/
│   ├── links.tex     ← URLs (PENDENTE = preencher depois)
│   ├── referencias.bib
│   ├── sbc-template.sty
│   └── sbc.bst
├── suporte/          ← figuras (par, ExReal, ExIA, gráficos)
└── anexos/
    ├── README.md
    └── poster.pdf    ← PENDENTE (copiar o poster aqui)
```

## Compilar

```bash
cd artigos/entrega-relatorio
./compilar.sh
```

## Links (`configs/links.tex`)

| Item | Status |
|------|--------|
| Dataset COCO | OK |
| Dataset pares (Drive) | OK |
| Checkpoint modelos (Drive) | OK |
| GitHub | **PENDENTE** |
| Google Presentation | **PENDENTE** |
| YouTube | **PENDENTE** |
| Poster PDF | **PENDENTE** → `anexos/poster.pdf` |

Quando tiver a URL, edite só `configs/links.tex` e recompile.
