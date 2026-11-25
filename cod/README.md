# Código — treino VerifAI (v3)

## Estrutura

| Item | Função |
|------|--------|
| `main.ipynb` | Notebook de treino / avaliação |
| `data.py` | Pares, letterbox, splits |
| `features.py` | FFT / HOG / LBP + augs |
| `svm.py` | ThunderSVM → sklearn RBF |
| `models/` | `.pkl` salvos |
| `local/` | cache `.npz` |
| `results/` | CSV / gráficos |
| `tentativas/` | v1, v2 e ferramentas antigas |

## Split

| Treino | Teste |
|--------|-------|
| 4000 pares | 1000 pares |

Rode `main.ipynb` e teste o modelo no app com `./run_app.sh`.
