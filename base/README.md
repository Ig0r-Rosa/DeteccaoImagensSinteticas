# Bases de dados (só local — não versionadas no git)

## Download (Drive)

Pacote com as bases do treino v3 (pares REAL × IA):

https://drive.google.com/file/d/1rCeLicFZm_cDrxg0qjpc82ROfdeRGwra/view?usp=sharing

Extraia em `base/` mantendo as pastas abaixo.

## Organização

```
base/
├── README.md
├── PARES_REAL_IA.md
├── landscape_real/   ← REAL (5k)
├── pares_ia/         ← FAKE (5k, mesma cena/pose)
├── pares_json/       ← prompts (opcional)
└── pares_preview/    ← amostras REAL|IA (opcional)
```

| Pasta | Rótulo | Papel |
|-------|--------|-------|
| `landscape_real/` | REAL (1) | Fotos reais (amostra COCO) |
| `pares_ia/` | FAKE (0) | IA gerada pareada (mesmo nome) |

Mesmo nome de arquivo (`00042.jpg`) = **par alinhado**.

## Treino no notebook

| Modo | Treino | Teste |
|------|--------|-------|
| `avaliacao` | 4000 pares | 1000 pares |
| `app` | 5000 pares | só no app |

Arquivo: `cod/main_v3.ipynb` · tag de cache `v3pares`.
