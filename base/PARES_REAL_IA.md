# Pares REAL × IA — concluído

5 000 pares gerados (REAL → JSON → IA com ControlNet Depth).

## Pastas

| Pasta | Conteúdo |
|-------|----------|
| `landscape_real/` | REAL |
| `pares_ia/` | FAKE pareada |
| `pares_json/` | JSON/prompt (opcional) |
| `pares_preview/` | Comparativos |

## Download

https://drive.google.com/file/d/1rCeLicFZm_cDrxg0qjpc82ROfdeRGwra/view?usp=sharing

## Treino

Use `cod/main_v3.ipynb`:

1. `MODO = "avaliacao"` → 4000/1000  
2. `MODO = "app"` → 5000 no treino (teste no VerifAI)

Ferramentas de geração (torch/diffusers/BLIP) foram removidas do `.venv` após concluir esta etapa.
