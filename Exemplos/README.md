# Exemplos e Apresentação

## Ferramenta usada nos testes

**[ZeroGPT AI Image Detector](https://www.zerogpt.com/pt/ai-image-detector)**

Usamos este detector online para gerar os screenshots das pastas `GeradaIA/` e `EditadaIA/`. Ele serve como **referência exploratória** — o detector final do trabalho será implementado por nós com visão clássica.

## Apresentação para o professor

Abrir no navegador:

```
Exemplos/apresentacao-ideia.html
```

- **13 slides** — ideia, cenários reais, visão clássica, pipeline, exemplos
- Fonte **Roboto** + cores Material Design
- Usa as imagens das pastas abaixo

## Pastas de exemplo

| Pasta | Conteúdo |
|---|---|
| `GeradaIA/` | Imagem sintética + testes com transformações (99% → 6% real) |
| `EditadaIA/` | Foto real + testes de resolução e transformação |

## Narrativa dos exemplos

**Cenários que motivam o trabalho:**
- Manipulação de imagens de políticos em redes sociais durante período eleitoral
- Manipulação de fotos de imóveis em anúncios para atrair compradores

**GeradaIA** — mesma imagem IA, transformações progressivas:

| Arquivo | Resultado mock |
|---|---|
| Teste1.png | 99% FALSA |
| Teste3.png | 93% FALSA |
| Teste4-FiltroQualidade.png | 63% FALSA |
| Teste5-Filtro+Screenshots.png | 6% REAL (enganou) |

**EditadaIA** — foto real:

| Arquivo | Resultado mock |
|---|---|
| TesteReducaoReso.png | 97% REAL |
| TestePosTransformacao.png | 99% FALSA (falso positivo) |
