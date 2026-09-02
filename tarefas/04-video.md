# Tarefas — Vídeo da Apresentação

## Resumo da entrega

| Item | Detalhe |
|---|---|
| **O que enviar** | MP4 H.265 (HEVC) + link do YouTube |
| **Resolução** | FHD 1920×1080 |
| **Conteúdo** | Apresentação dos slides + demo do código |
| **Tela final** | URL do YouTube |
| **Prazo** | 30/11/2026 |

## Pré-requisitos

- Slides prontos ([03-slides.md](03-slides.md)).
- Código funcional com demo gravável ([02-codigo.md](02-codigo.md) Fase 7).

---

## Fase 1 — Roteiro

| # | Tarefa | Resp. | Critério de pronto |
|---|---|---|---|
| 1.1 | Escrever roteiro (5–8 minutos) | B | Tempo estimado por seção |
| 1.2 | Definir quem fala cada parte | A+B | Divisão clara |
| 1.3 | Planejar demo ao vivo (classificar 2–3 imagens com transformações) | B | Script da demo escrito |

**Estrutura sugerida do roteiro:**

| Trecho | Duração | Conteúdo |
|---|---|---|
| Abertura | ~30 s | Título, autores, problema |
| Contexto | ~1 min | Política eleitoral, imóveis, por que robustez importa |
| Método | ~2 min | Dataset, transformações, features clássicas, SVM |
| Demo | ~2 min | Extrair features e classificar imagens ao vivo |
| Resultados | ~1,5 min | Gráfico de degradação |
| Conclusão | ~1 min | Achados e limitações |

- [ ] 1.1 — Roteiro **(B)**
- [ ] 1.2 — Divisão de falas **(A+B)**
- [ ] 1.3 — Script da demo **(B)**

---

## Fase 2 — Gravação

| # | Tarefa | Resp. | Critério de pronto |
|---|---|---|---|
| 2.1 | Escolher ferramenta (OBS, Zoom, QuickTime, etc.) | B | Grava tela + áudio |
| 2.2 | Testar áudio e resolução FHD | B | Sem ruído excessivo |
| 2.3 | Gravar apresentação dos slides | A+B | 1–2 takes aceitáveis |
| 2.4 | Gravar demo do código (terminal + resultado) | B | Comando roda sem erro |
| 2.5 | Gravar tela final com URL do YouTube (placeholder OK na 1ª gravação) | B | Texto legível |

- [ ] 2.1 — Ferramenta **(B)**
- [ ] 2.2 — Teste de áudio/vídeo **(B)**
- [ ] 2.3 — Gravação dos slides **(A+B)**
- [ ] 2.4 — Demo do código **(B)**
- [ ] 2.5 — Tela final **(B)**

**Entregável parcial:** gravação bruta completa.

---

## Fase 3 — Edição e publicação

| # | Tarefa | Resp. | Critério de pronto |
|---|---|---|---|
| 3.1 | Editar vídeo (cortar erros, juntar takes) | B | Fluxo contínuo |
| 3.2 | Exportar em MP4 FHD | B | 1920×1080 |
| 3.3 | Converter para H.265 se necessário: `ffmpeg -i input.mp4 -c:v libx265 -crf 28 output.mp4` | B | Codec HEVC confirmado |
| 3.4 | Upload no YouTube (público ou não listado) | B | Link gerado |
| 3.5 | Atualizar tela final com URL real e re-exportar se necessário | B | URL correta no vídeo |
| 3.6 | Enviar MP4 + link do YouTube no Moodle | A+B | Entrega confirmada |

- [ ] 3.1 — Edição **(B)**
- [ ] 3.2 — Export FHD **(B)**
- [ ] 3.3 — Conversão H.265 **(B)**
- [ ] 3.4 — Upload YouTube **(B)**
- [ ] 3.5 — Tela final com URL **(B)**
- [ ] 3.6 — Enviar no Moodle **(A+B)**

**Entregável final:** MP4 H.265 + link YouTube no Moodle.
