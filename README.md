# Detecção de Imagens Sintéticas após Transformações

Trabalho final — **INE410159 / TRV410001 — Visão Computacional** (UFSC, 2026.2).

**Autores:** Igor de Matos da Rosa · Nathália Geraldino Ribas  
**Repositório:** https://github.com/Ig0r-Rosa/DeteccaoImagensSinteticas

---

## Objetivo

Medir a **robustez** de um detector clássico (FFT / HOG / LBP + SVM) quando imagens reais e sintéticas sofrem transformações típicas de redes sociais (JPEG, crop, resize, blur, etc.).

A solução final é o **VerifAI**: app Streamlit que analisa a imagem inteira e devolve a chance de conteúdo sintético (modelo de **Fusão**).

---

## Estrutura

```
DeteccaoImagensSinteticas/
├── app.py / app/          # VerifAI (Streamlit + detector)
├── cod/
│   ├── main.ipynb         # treino / avaliação
│   ├── data.py            # pares + letterbox
│   ├── features.py        # FFT / HOG / LBP
│   ├── svm.py             # treino SVM
│   ├── models/            # checkpoints (.pkl) — local / Drive
│   ├── results/           # CSV e gráficos
│   └── tentativas/        # notebooks v1 e v2
├── base/                  # datasets (local / Drive)
├── apresentacao/          # slides, poster e vídeo
├── artigos/               # entregas LaTeX + relatório
├── exemplos/              # testes exploratórios
├── requirements.txt
└── run_app.sh
```

| Pasta | Papel |
|-------|--------|
| `app/` + `app.py` | Interface e inferência |
| `cod/` | Treino modular + resultados |
| `base/` | Pares REAL × IA e pasta `teste_app/` |
| `apresentacao/` | PDF dos slides, poster e vídeo |
| `artigos/` | Relatório e entregas intermediárias |

---

## Links úteis

| Item | Link |
|------|------|
| Dataset (pares) | https://drive.google.com/file/d/1rCeLicFZm_cDrxg0qjpc82ROfdeRGwra/view?usp=sharing |
| Checkpoints (`.pkl`) | https://drive.google.com/file/d/1j8q-wCwH06oZBbQOCZ4EIS84k2nz5ry0/view?usp=sharing |
| GitHub | https://github.com/Ig0r-Rosa/DeteccaoImagensSinteticas |

Slides, poster e vídeo da entrega ficam em `apresentacao/`.

---

## Bases

Extraia o pacote do Drive em `base/`:

| Pasta | Conteúdo |
|-------|----------|
| `landscape_real/` | REAL (5k) |
| `pares_ia/` | FAKE pareada (5k) |
| `teste_app/` | 5 REAL + 5 IA × variantes (teste manual no app) |

Mesmo nome de arquivo = mesmo par. Detalhes: [`base/README.md`](base/README.md).

---

## Como rodar

### 1. Ambiente

```bash
python -m pip install -r requirements.txt
```

Use o `.venv` do projeto (ThunderSVM). Coloque os `.pkl` em `cod/models/` (Drive acima).

### 2. Treino

| Item | Valor |
|------|--------|
| Notebook | `cod/main.ipynb` |
| Split | **4000** treino / **1000** teste |
| Entrada | Imagem inteira (letterbox 224; HOG 128) |
| Saída | `cod/models/svm_v3_*.pkl` + scalers |

### 3. App (VerifAI)

```bash
./run_app.sh
```

- Veredito padrão = score da **Fusão**
- Faixa **40–60%** → inconclusivo
- FFT / HOG / LBP disponíveis nas configurações

Mais detalhes: [`cod/README.md`](cod/README.md).

---

## Artigos

| Pasta | Conteúdo |
|-------|----------|
| `artigos/entrega-01` … `03` | Entregas intermediárias |
| `artigos/entrega-relatorio` | Relatório final |

```bash
cd artigos/entrega-relatorio && ./compilar.sh
```

---

## Resultados principais

| Métrica | Valor |
|---------|--------|
| Acurácia da fusão (holdout) | **88,15%** |
| JPEG / resize / blur | em geral **> 87%** |
| Crop forte / pipeline | maiores quedas (~74–80%) |
