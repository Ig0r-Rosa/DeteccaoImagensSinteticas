# Tarefas — Código da Solução

## Resumo da entrega

| Item | Detalhe |
|---|---|
| **O que enviar** | ZIP do repositório GitHub |
| **Extras** | Modelos treinados em `models/` (`.pkl`) + instruções no README |
| **Prazo** | 30/11/2026 |
| **Regra IAGen** | Se usar IA, registrar em `IAGEN.md` com prompts e explicação |

## Pré-requisitos

- Nenhum (esta é a primeira etapa do projeto).

---

## Fase 1 — Setup do projeto

| # | Tarefa | Resp. | Critério de pronto |
|---|---|---|---|
| 1.1 | Criar repositório no GitHub | B | Repo público criado |
| 1.2 | Definir estrutura de pastas (`src/`, `data/`, `notebooks/`, `results/`) | B | Pastas criadas e documentadas no README |
| 1.3 | Criar `requirements.txt` (OpenCV, scikit-learn, scikit-image, NumPy, SciPy, matplotlib) | B | `pip install -r requirements.txt` funciona |
| 1.4 | Configurar ambiente virtual e testar importações | A | Sem erros de import |
| 1.5 | Adicionar `.gitignore` (data/, models/, `__pycache__/`) | B | Arquivos grandes não versionados |

- [ ] 1.1 — Criar repositório no GitHub **(B)**
- [ ] 1.2 — Estrutura de pastas **(B)**
- [ ] 1.3 — `requirements.txt` **(B)**
- [ ] 1.4 — Ambiente virtual **(A)**
- [ ] 1.5 — `.gitignore` **(B)**

**Entregável parcial:** repositório clonável com estrutura básica.

---

## Fase 2 — Fundamentação e dados

| # | Tarefa | Resp. | Critério de pronto |
|---|---|---|---|
| 2.1 | Ler 3–5 papers + testar imagens no [ZeroGPT AI Image Detector](https://www.zerogpt.com/pt/ai-image-detector) | A+B | Resumo em `docs/referencias.md` + screenshots em `Exemplos/` |
| 2.2 | Escolher dataset (ex.: CIFAKE, GenImage subset, ou real vs Stable Diffusion) | A | Dataset definido com justificativa |
| 2.3 | Baixar e organizar dataset em `data/raw/` (real/ e fake/) | A | Pastas com imagens separadas por classe |
| 2.4 | Criar split treino/val/teste (ex.: 70/15/15) | A | Arquivo `data/splits.json` gerado |
| 2.5 | Validar balanceamento de classes e tamanho mínimo do dataset | B | Tabela com contagem por split |

- [ ] 2.1 — Revisão bibliográfica **(A+B)**
- [ ] 2.2 — Escolha do dataset **(A)**
- [ ] 2.3 — Download e organização **(A)**
- [ ] 2.4 — Split treino/val/teste **(A)**
- [ ] 2.5 — Validação do dataset **(B)**

**Entregável parcial:** dataset pronto para treino e experimentos.

---

## Fase 3 — Pipeline de transformações

| # | Tarefa | Resp. | Critério de pronto |
|---|---|---|---|
| 3.1 | Criar `src/transforms/jpeg.py` (qualidades 90, 70, 50) | A | Função salva JPEG e retorna imagem |
| 3.2 | Criar `src/transforms/geometry.py` (resize, crop 10/25/40%) | A | Funções testadas com 1 imagem |
| 3.3 | Criar `src/transforms/filters.py` (blur, sharpen, brilho/contraste) | A | Funções testadas com 1 imagem |
| 3.4 | Criar `src/transforms/screenshot.py` (simular captura de tela) | A | Resultado visualmente plausível |
| 3.5 | Criar `src/transforms/pipeline.py` (crop → resize → JPEG) | A | Pipeline combinado funciona |
| 3.6 | Criar script `scripts/apply_transforms.py` (aplica todas em lote) | A | Gera pasta `data/transformed/` |
| 3.7 | Gerar amostras visuais antes/depois para o poster | A | 3–5 pares de imagens em `results/samples/` |

- [ ] 3.1 — JPEG **(A)**
- [ ] 3.2 — Geometria **(A)**
- [ ] 3.3 — Filtros **(A)**
- [ ] 3.4 — Screenshot **(A)**
- [ ] 3.5 — Pipeline combinado **(A)**
- [ ] 3.6 — Script em lote **(A)**
- [ ] 3.7 — Amostras visuais **(A)**

**Entregável parcial:** todas as transformações aplicáveis em lote.

---

## Fase 4 — Extração de features clássicas

| # | Tarefa | Resp. | Critério de pronto |
|---|---|---|---|
| 4.1 | Criar `src/features/fft.py` (espectro de magnitude — domínio frequência) | B | Vetor de features por imagem |
| 4.2 | Criar `src/features/hog.py` (HOG — domínio espaço) | B | Vetor de features por imagem |
| 4.3 | Criar `src/features/texture.py` (LBP ou GLCM — domínio valor/textura) | B | Vetor de features por imagem |
| 4.4 | Criar `src/features/histogram.py` (histograma RGB/HSV) | B | Vetor de features por imagem |
| 4.5 | Criar `src/features/extract.py` (unifica extração por tipo) | B | `extract(image, type)` funciona |
| 4.6 | Testar extração em 5 imagens reais + 5 sintéticas | B | Features distintas visualizáveis |
| 4.7 | Documentar cada feature em `docs/metodo.md` (domínio + justificativa) | B | Tabela domínio → feature |

- [ ] 4.1 — FFT **(B)**
- [ ] 4.2 — HOG **(B)**
- [ ] 4.3 — LBP/GLCM **(B)**
- [ ] 4.4 — Histograma **(B)**
- [ ] 4.5 — Unificador **(B)**
- [ ] 4.6 — Teste visual **(B)**
- [ ] 4.7 — Documentação **(B)**

**Entregável parcial:** extratores clássicos funcionando.

---

## Fase 5 — Classificador e treino

| # | Tarefa | Resp. | Critério de pronto |
|---|---|---|---|
| 5.1 | Escolher classificador principal (SVM ou kNN) | B | Decisão documentada em `docs/metodo.md` |
| 5.2 | Criar `src/build_features.py` (gera matriz features + labels do split) | B | Arquivo `.npz` ou `.csv` gerado |
| 5.3 | Criar `src/train.py` (treina e valida com cross-validation) | B | Acurácia no val registrada |
| 5.4 | Salvar modelos em `models/` (ex.: `svm_fft.pkl`, `svm_hog.pkl`) | B | Modelos carregáveis com joblib |
| 5.5 | Normalizar features (StandardScaler) antes do treino | B | Scaler salvo junto ao modelo |
| 5.6 | *(Opcional)* Treinar 2ª família de feature para comparação | B | Dois modelos comparáveis |
| 5.7 | Documentar hiperparâmetros (C, kernel, k) em `docs/metodo.md` | B | Tabela preenchida |

- [ ] 5.1 — Escolha do classificador **(B)**
- [ ] 5.2 — Matriz de features **(B)**
- [ ] 5.3 — Treino **(B)**
- [ ] 5.4 — Salvar modelos **(B)**
- [ ] 5.5 — Normalização **(B)**
- [ ] 5.6 — Segunda família (opcional) **(B)**
- [ ] 5.7 — Hiperparâmetros **(B)**

**Entregável parcial:** classificador(es) treinado(s) e salvos.

---

## Fase 6 — Avaliação e resultados

| # | Tarefa | Resp. | Critério de pronto |
|---|---|---|---|
| 6.1 | Criar `src/evaluate.py` (accuracy, F1 por transformação e por feature) | B | Métricas calculadas corretamente |
| 6.2 | Avaliar baseline em imagens **originais** | B | Resultado registrado |
| 6.3 | Avaliar cada transformação isoladamente | B | Tabela com degradação por tipo |
| 6.4 | Avaliar pipeline combinado | B | Resultado registrado |
| 6.5 | Comparar qual família de feature degrada mais (FFT vs HOG vs LBP) | A | Tabela comparativa |
| 6.6 | Gerar gráfico de degradação (`results/degradation_chart.png`) | A | Gráfico legível para poster/slides |
| 6.7 | Exportar resultados em `results/metrics.csv` e `results/metrics.json` | A | Arquivos versionados no repo |
| 6.8 | Escrever análise: qual padrão visual some com cada transformação | A+B | Texto em `docs/resultados.md` |

- [ ] 6.1 — Script de avaliação **(B)**
- [ ] 6.2 — Baseline original **(B)**
- [ ] 6.3 — Transformações isoladas **(B)**
- [ ] 6.4 — Pipeline combinado **(B)**
- [ ] 6.5 — Comparação entre features **(A)**
- [ ] 6.6 — Gráfico de degradação **(A)**
- [ ] 6.7 — Export CSV/JSON **(A)**
- [ ] 6.8 — Análise de padrões **(A+B)**

**Entregável parcial:** tabela e gráfico de degradação prontos.

---

## Fase 7 — Finalização e entrega

| # | Tarefa | Resp. | Critério de pronto |
|---|---|---|---|
| 7.1 | Atualizar README com instruções (`extract`, `train`, `evaluate`) | B | Outra pessoa consegue rodar seguindo o README |
| 7.2 | Documentar local dos modelos em `models/` no README | B | Caminhos explícitos |
| 7.3 | Criar `IAGEN.md` se usou IA generativa (prompts + explicação) | A+B | Conforme regra do professor |
| 7.4 | Revisão cruzada de todo o código | A+B | Sem bugs bloqueantes |
| 7.5 | Baixar repo do GitHub e gerar ZIP | B | Arquivo `.zip` pronto para Moodle |
| 7.6 | Enviar ZIP + indicar modelos treinados no Moodle | A+B | Entrega confirmada |

- [ ] 7.1 — README técnico **(B)**
- [ ] 7.2 — Documentação dos modelos **(B)**
- [ ] 7.3 — `IAGEN.md` **(A+B)**
- [ ] 7.4 — Revisão cruzada **(A+B)**
- [ ] 7.5 — Gerar ZIP **(B)**
- [ ] 7.6 — Enviar no Moodle **(A+B)**

**Entregável final:** ZIP no Moodle com pipeline clássico funcional e modelos documentados.
