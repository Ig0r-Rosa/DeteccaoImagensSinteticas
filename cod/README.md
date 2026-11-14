# Código — Visão clássica + SVM

## Notebooks

| Arquivo | Versão | Uso |
|---------|--------|-----|
| `main_v3.ipynb` | **Atual** | Instagram AI × fotos reais (treino do app) |
| `main_v2_robusto.ipynb` | Legado | Robustez a transformações (CIFAKE/era anterior) |
| `main_v1.ipynb` | Legado | Primeiro pipeline |

## Pacote `v3/`

| Módulo | Função |
|--------|--------|
| `data.py` | Carrega bases, letterbox, splits |
| `features.py` | FFT / HOG / LBP + augs |
| `svm_gpu.py` | ThunderSVM → sklearn RBF |

## Pastas geradas (não versionar)

| Pasta | Conteúdo |
|-------|----------|
| `models/` | `svm_v3_*.pkl`, `scaler_v3_*.pkl` |
| `local/` | cache de features `.npz` |
| `results/` | CSV e gráficos |

## Docs

- `docs/explicacao-main.md` — explicação didática do pipeline

## Como treinar (v3)

1. Bases em `../base/` (ver `base/README.md`)
2. Abrir `main_v3.ipynb` no kernel do `.venv`
3. `MODO = "completo"` e rodar todas as células
