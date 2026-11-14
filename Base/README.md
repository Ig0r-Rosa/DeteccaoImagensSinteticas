# Bases de dados (só local — não versionadas no git)

Baixe e organize **nesta pasta**. As imagens não entram no repositório.

## Organização das pastas

```
Base/
├── README.md              ← este arquivo
├── instagram_ai/          ← FAKE (IA de redes sociais)
├── celeba_real/           ← REAL (rostos)
├── landscape_real/        ← REAL (cenas / paisagens / dia a dia)
└── archive/               ← CIFAKE (opcional; o treino v3 atual NÃO usa)
```

| Pasta | Rótulo | Conteúdo |
|-------|--------|----------|
| `instagram_ai/` | FAKE (0) | Imagens sintéticas de Instagram/Twitter |
| `celeba_real/` | REAL (1) | Retratos reais (CelebA) |
| `landscape_real/` | REAL (1) | Cenas reais diversas (recomendado para o app) |
| `archive/` | — | CIFAKE 32×32 — **ignorada** no treino v3 social |

Subpastas dentro de cada pasta são aceitas (o loader faz busca recursiva).

## Links para download

### FAKE — Harvard Dataverse (Instagram / Twitter AI)

- **Link:** https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/MGJPBL  
- Extraia as imagens para `Base/instagram_ai/`.

### REAL — CelebA (rostos)

- **Link:** https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html  
- Use `img_align_celeba.zip` (~1,4 GB).  
- Extraia as fotos para `Base/celeba_real/` (ex.: arquivos `.jpg` direto ou em subpasta).

### REAL — cenas (landscape / COCO / etc.)

O CelebA cobre só rostos. Para o app não marcar foto de parque/rua como IA, coloque cenas variadas em `Base/landscape_real/`:

- LHQ: https://github.com/universome/alis  
- Ou amostras de COCO / Open Images / Unsplash

## Treino

O notebook `Cod/main_v3_instagram.ipynb` lê só `instagram_ai` + `celeba_real` + `landscape_real`, com a **imagem inteira** (letterbox 224×224).
