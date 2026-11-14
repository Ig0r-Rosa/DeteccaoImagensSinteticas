# Artigos

Cada entrega fica em **uma pasta**. Na raiz da pasta só entram o fonte, o PDF e o script de build.

```
artigos/
├── entrega-01/
│   ├── main.tex
│   ├── main.pdf
│   ├── compilar.sh
│   ├── configs/      ← .sty .bst .bib
│   └── suporte/      ← imagens/HTML auxiliares
├── entrega-02/
└── entrega-02-revisao/
```

| Pasta | Conteúdo |
|-------|----------|
| `entrega-01/` | Relatório / artigo inicial |
| `entrega-02/` | Segunda versão |
| `entrega-02-revisao/` | Revisão da entrega 02 |

## Compilar

```bash
cd artigos/entrega-02-revisao
./compilar.sh
```

Gera/atualiza `main.pdf` e limpa `.aux`, `.log`, etc.
