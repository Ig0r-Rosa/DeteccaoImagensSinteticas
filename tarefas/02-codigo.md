# Tarefas — Código da Solução

## Resumo da entrega

| Item | Detalhe |
|---|---|
| **O que enviar** | ZIP do repositório GitHub |
| **Extras** | Link para checkpoints da rede neural |
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
| 1.3 | Criar `requirements.txt` (PyTorch, OpenCV, Pillow, scikit-learn, matplotlib) | B | `pip install -r requirements.txt` funciona |
| 1.4 | Configurar ambiente virtual e testar importações | A | Sem erros de import |
| 1.5 | Adicionar `.gitignore` (data/, checkpoints/, `__pycache__/`) | B | Arquivos grandes não versionados |

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
| 2.1 | Ler 3–5 papers sobre detecção de imagens sintéticas e robustez | A+B | Resumo de 1 parágrafo por paper em `docs/referencias.md` |
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

## Fase 4 — Detector e treino

| # | Tarefa | Resp. | Critério de pronto |
|---|---|---|---|
| 4.1 | Escolher abordagem (CNN simples, ResNet fine-tune ou pré-treinado) | B | Decisão documentada em `docs/metodo.md` |
| 4.2 | Criar `src/dataset.py` (DataLoader com imagens originais) | B | Batch carrega sem erro |
| 4.3 | Criar `src/model.py` (arquitetura do classificador) | B | Forward pass OK |
| 4.4 | Criar `src/train.py` (loop de treino com val loss) | B | Treino roda até convergir |
| 4.5 | Salvar checkpoint em `checkpoints/best_model.pt` | B | Arquivo existe e é carregável |
| 4.6 | *(Opcional)* Segundo modelo (frequência ou CLIP/ViT) | B | Comparável com baseline |
| 4.7 | Documentar hiperparâmetros e tempo de treino | B | Tabela em `docs/metodo.md` |

- [ ] 4.1 — Escolha da abordagem **(B)**
- [ ] 4.2 — DataLoader **(B)**
- [ ] 4.3 — Modelo **(B)**
- [ ] 4.4 — Treino **(B)**
- [ ] 4.5 — Checkpoint **(B)**
- [ ] 4.6 — Modelo opcional **(B)**
- [ ] 4.7 — Documentação de hiperparâmetros **(B)**

**Entregável parcial:** modelo treinado com checkpoint salvo.

---

## Fase 5 — Avaliação e resultados

| # | Tarefa | Resp. | Critério de pronto |
|---|---|---|---|
| 5.1 | Criar `src/evaluate.py` (accuracy, F1, AUC por transformação) | B | Métricas calculadas corretamente |
| 5.2 | Avaliar baseline em imagens **originais** | B | Resultado registrado |
| 5.3 | Avaliar cada transformação isoladamente | B | Tabela com degradação por tipo |
| 5.4 | Avaliar pipeline combinado | B | Resultado registrado |
| 5.5 | Gerar gráfico de degradação (`results/degradation_chart.png`) | A | Gráfico legível para poster/slides |
| 5.6 | Exportar resultados em `results/metrics.csv` e `results/metrics.json` | A | Arquivos versionados no repo |
| 5.7 | Escrever análise inicial dos resultados (1 página) | A+B | Texto em `docs/resultados.md` |

- [ ] 5.1 — Script de avaliação **(B)**
- [ ] 5.2 — Baseline original **(B)**
- [ ] 5.3 — Transformações isoladas **(B)**
- [ ] 5.4 — Pipeline combinado **(B)**
- [ ] 5.5 — Gráfico de degradação **(A)**
- [ ] 5.6 — Export CSV/JSON **(A)**
- [ ] 5.7 — Análise inicial **(A+B)**

**Entregável parcial:** tabela e gráfico de degradação prontos.

---

## Fase 6 — Finalização e entrega

| # | Tarefa | Resp. | Critério de pronto |
|---|---|---|---|
| 6.1 | Atualizar README com instruções de execução (`train`, `evaluate`) | B | Outra pessoa consegue rodar seguindo o README |
| 6.2 | Documentar local dos checkpoints no README | B | Link ou caminho explícito |
| 6.3 | Criar `IAGEN.md` se usou IA generativa (prompts + explicação) | A+B | Conforme regra do professor |
| 6.4 | Revisão cruzada de todo o código | A+B | Sem bugs bloqueantes |
| 6.5 | Baixar repo do GitHub e gerar ZIP | B | Arquivo `.zip` pronto para Moodle |
| 6.6 | Enviar ZIP + link dos checkpoints no Moodle | A+B | Entrega confirmada |

- [ ] 6.1 — README técnico **(B)**
- [ ] 6.2 — Link dos checkpoints **(B)**
- [ ] 6.3 — `IAGEN.md` **(A+B)**
- [ ] 6.4 — Revisão cruzada **(A+B)**
- [ ] 6.5 — Gerar ZIP **(B)**
- [ ] 6.6 — Enviar no Moodle **(A+B)**

**Entregável final:** ZIP no Moodle com código funcional e checkpoints documentados.
