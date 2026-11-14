# Pares REAL × IA (com retomada)

```
foto REAL  →  JSON (BLIP)  →  imagem IA (SD1.5 + ControlNet Depth)
```

## Pastas

| Pasta | Conteúdo |
|-------|----------|
| `base/landscape_real/` | Fotos REAL (fonte) |
| `base/pares_json/` | JSON/prompt por imagem |
| `base/pares_ia/` | Versão IA (mesmo stem) |
| `base/pares_preview/` | REAL\|IA lado a lado (amostra) |
| `base/pares_estado.json` | Progresso / erros |

## Retomada

- Se o `.json` já existe → pula caption.
- Se o `.jpg` em `pares_ia/` já existe → pula geração.
- Pode dar Ctrl+C e rodar de novo: **continua de onde parou**.

## Comandos

```bash
# Teste (5 imagens)
./run_pares.sh --limite 5

# Batch completo (retomável)
./run_pares.sh

# Só JSON ou só IA
./run_pares.sh --so-caption
./run_pares.sh --so-gerar
```

## Modelos locais

| Etapa | Modelo |
|-------|--------|
| Caption | `Salesforce/blip-image-captioning-large` |
| Geração | `sd-legacy/stable-diffusion-v1-5` + `lllyasviel/sd-controlnet-depth` |

Baixados automaticamente na 1ª execução (cache Hugging Face).

## Hardware

RTX 3050 6GB · resolução 512×512 · ~20 steps. Demora por imagem; as 5k levam **muitas horas**.
