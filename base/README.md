# Bases de dados (só local — não versionadas no git)

## Organização

```
base/
├── README.md
├── PARES_REAL_IA.md   ← ideia: REAL → JSON → IA (pares alinhados)
├── instagram_ai/      ← FAKE (~60k) — IA estilo rede social (Harvard)
└── landscape_real/    ← REAL — 5k fotos COCO (amostra válida)
```

| Pasta | Rótulo | Papel |
|-------|--------|-------|
| `instagram_ai/` | FAKE (0) | IA (Harvard Dataverse) |
| `landscape_real/` | REAL (1) | 5k fotos reais (amostra do COCO 2017) |

> **CelebA / CIFAKE** não entram no treino v3.

## REAL atual

- **5 000** JPGs aleatórias, legíveis, em `landscape_real/` (plana).
- Origem: COCO 2017 `train2017` (resto removido para economizar disco).

## Próximo passo (recomendado)

Gerar **pares IA** com a mesma cena/pose a partir dessas 5k — ver **`PARES_REAL_IA.md`**.

Stack local: **ComfyUI + Florence-2 + FLUX Nunchaku (INT4) + ControlNet Depth**.

## FAKE — Harvard Dataverse

- https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/MGJPBL  
- Pasta: `base/instagram_ai/`

## Alternativa in-the-wild (aguardando acesso)

- [ITW-SM](https://huggingface.co/datasets/dkarageo/itw-sm) — 5k REAL + 5k IA já de Instagram/FB/X/LinkedIn.
