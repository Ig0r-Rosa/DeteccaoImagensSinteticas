# Detecção de Imagens Sintéticas após Transformações

Trabalho final — **INE410159 / TRV410001 — Visão Computacional** (UFSC, 2026.2).

**Autores:** Igor de Matos da Rosa · Nathália Geraldino Ribas

---

## Objetivo

Medir a **robustez** de um detector clássico (FFT / HOG / LBP + SVM) quando imagens reais e sintéticas sofrem transformações típicas de redes sociais.

## Estrutura do repositório

```
DeteccaoImagensSinteticas/
├── README.md
├── requirements.txt
├── run_app.sh              # sobe o app no .venv
├── app.py                  # Streamlit (VerifAI)
├── app/                    # inferência (detector)
├── cod/                    # notebooks + pacote v3 + models
├── base/                   # datasets (local; ver base/README.md)
├── artigos/                # 1 pasta por entrega (.tex + .pdf + compilar.sh)
├── docs/                   # entregas Moodle + checklists
├── exemplos/               # screenshots ZeroGPT + HTMLs
└── referencias/            # PDFs de papers
```

| Pasta | Papel |
|-------|--------|
| `cod/` | Treino (`main.ipynb`, `data.py`, `features.py`, `svm.py`) |
| `app/` + `app.py` | Interface Streamlit |
| `base/` | Imagens de treino (não versionadas) |
| `artigos/` | Relatórios LaTeX |
| `docs/` | Entregas e tarefas |
| `exemplos/` | Material exploratório |
| `referencias/` | Artigos de apoio |

## Bases (pares REAL × IA)

Download (Drive):  
https://drive.google.com/file/d/1rCeLicFZm_cDrxg0qjpc82ROfdeRGwra/view?usp=sharing

| Pasta | Papel |
|-------|--------|
| `base/landscape_real/` | REAL (5k) |
| `base/pares_ia/` | FAKE pareada (5k) |

Detalhes: [`base/README.md`](base/README.md).

## Treino (v3) e app

| Passo | Comando / arquivo |
|-------|-------------------|
| Bases | Drive acima → `base/` |
| Notebook | `cod/main.ipynb` (kernel do `.venv`) |
| Split | 4000 treino / 1000 teste |
| Modelos | `cod/models/svm_v3_*.pkl` |
| App | `./run_app.sh` |

- Entrada: **imagem inteira** (letterbox 224; HOG 128).
- Veredito do app: score da **Fusão**.
- Faixa **40–60%** → inconclusivo.

```bash
python -m pip install -r requirements.txt
./run_app.sh
```

## Artigos

```bash
cd artigos/entrega-02-revisao && ./compilar.sh
```

Detalhes: [`artigos/README.md`](artigos/README.md).

## Entregas Moodle

Prazo: **30/11/2026** · [`docs/entregas.md`](docs/entregas.md) · checklists em [`docs/tarefas/`](docs/tarefas/).

| # | Entrega | Checklist |
|---|---------|-----------|
| 1 | Relatório | [`docs/tarefas/01-relatorio.md`](docs/tarefas/01-relatorio.md) |
| 2 | Código | [`docs/tarefas/02-codigo.md`](docs/tarefas/02-codigo.md) |
| 3 | Slides | [`docs/tarefas/03-slides.md`](docs/tarefas/03-slides.md) |
| 4 | Vídeo | [`docs/tarefas/04-video.md`](docs/tarefas/04-video.md) |
| 5 | Poster | [`docs/tarefas/05-poster.md`](docs/tarefas/05-poster.md) |

## Abordagem

Features clássicas → SVM → medir queda de acurácia por transformação (JPEG, crop, resize, blur, pipeline).

Docs do código: [`cod/README.md`](cod/README.md) · [`cod/docs/explicacao-main.md`](cod/docs/explicacao-main.md).

## Regras de IAGen

Se usar Copilot/Cursor/ChatGPT: declarar ferramenta e prompts em `docs/iagen.md` (criar quando aplicável).
