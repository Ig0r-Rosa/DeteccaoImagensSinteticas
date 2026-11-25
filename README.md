# Detecção de Imagens Sintéticas após Transformações

Trabalho final — **INE410159 / TRV410001 — Visão Computacional** (UFSC, 2026.2).

**Autores:** Igor de Matos da Rosa · Nathália Geraldino Ribas

---

## Objetivo

Medir a **robustez** de um detector clássico (FFT / HOG / LBP + SVM) quando imagens reais e sintéticas sofrem transformações típicas de redes sociais (JPEG, crop, resize, blur, etc.).

---

## Estrutura

```
DeteccaoImagensSinteticas/
├── app.py                 # Streamlit (VerifAI)
├── app/                   # UI + detector
├── cod/
│   ├── main.ipynb         # treino / avaliação
│   ├── data.py            # pares + letterbox
│   ├── features.py        # FFT / HOG / LBP
│   ├── svm.py             # treino SVM
│   ├── models/            # .pkl
│   ├── results/           # CSV e gráficos
│   └── tentativas/        # v1, v2 e geração antiga
├── base/                  # datasets (local)
├── artigos/               # relatórios LaTeX
├── docs/                  # entregas Moodle
├── exemplos/
├── referencias/
├── requirements.txt
└── run_app.sh
```

| Pasta | Papel |
|-------|--------|
| `cod/` | Treino modular + modelos |
| `app/` + `app.py` | Interface VerifAI |
| `base/` | Pares REAL × IA (não versionados) |
| `artigos/` | Relatórios LaTeX |
| `docs/` | Entregas e checklists |

---

## Bases (pares REAL × IA)

Download:  
https://drive.google.com/file/d/1rCeLicFZm_cDrxg0qjpc82ROfdeRGwra/view?usp=sharing

| Pasta | Conteúdo |
|-------|----------|
| `base/landscape_real/` | REAL (5k) |
| `base/pares_ia/` | FAKE pareada (5k) |

Mesmo nome de arquivo = mesmo par. Detalhes: [`base/README.md`](base/README.md).

---

## Como rodar

### 1. Ambiente

```bash
python -m pip install -r requirements.txt
```

Use o `.venv` do projeto (ThunderSVM).

### 2. Treino

| Item | Valor |
|------|--------|
| Notebook | `cod/main.ipynb` |
| Split | **4000** treino / **1000** teste |
| Entrada | Imagem inteira (letterbox 224; HOG 128) |
| Saída | `cod/models/svm_v3_*.pkl` + scalers |

O app usa o modelo de **Fusão** (FFT+HOG+LBP).

### 3. App (VerifAI)

```bash
./run_app.sh
```

- Score da Fusão = chance de IA  
- Faixa **40–60%** → inconclusivo  
- Também dá para comparar Só FFT / HOG / LBP nas configurações  

Mais detalhes do código: [`cod/README.md`](cod/README.md).

---

## Artigos

```bash
cd artigos/entrega-03 && ./compilar.sh
```

Índice: [`artigos/README.md`](artigos/README.md).

---

## Entregas Moodle

Prazo: **30/11/2026** · [`docs/entregas.md`](docs/entregas.md)

| # | Entrega | Checklist |
|---|---------|-----------|
| 1 | Relatório | [`docs/tarefas/01-relatorio.md`](docs/tarefas/01-relatorio.md) |
| 2 | Código | [`docs/tarefas/02-codigo.md`](docs/tarefas/02-codigo.md) |
| 3 | Slides | [`docs/tarefas/03-slides.md`](docs/tarefas/03-slides.md) |
| 4 | Vídeo | [`docs/tarefas/04-video.md`](docs/tarefas/04-video.md) |
| 5 | Poster | [`docs/tarefas/05-poster.md`](docs/tarefas/05-poster.md) |
