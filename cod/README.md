# Código — Visão clássica + SVM

## Notebooks

| Arquivo | Versão | Uso |
|---------|--------|-----|
| `main_v3.ipynb` | **Atual** | Pares REAL × IA (`landscape_real` + `pares_ia`) |
| `main_v2_robusto.ipynb` | Legado | Robustez (CIFAKE) |
| `main_v1.ipynb` | Legado | Primeiro pipeline |

## Modos do v3

| `MODO` | Treino | Teste |
|--------|--------|-------|
| `avaliacao` | 4000 pares | 1000 pares |
| `app` | 5000 pares | só no VerifAI |

1. Rode `avaliacao` e confira a Fusão.  
2. Rode `app` e teste no `./run_app.sh`.

## Pacote `v3/`

| Módulo | Função |
|--------|--------|
| `data.py` | Pares, letterbox, splits |
| `features.py` | FFT / HOG / LBP + augs |
| `svm_gpu.py` | ThunderSVM → sklearn RBF |

## Pastas geradas

| Pasta | Conteúdo |
|-------|----------|
| `models/` | `svm_v3_*.pkl`, scalers |
| `local/` | cache `.npz` (`v3pares`) |
| `results/` | CSV / gráficos |
