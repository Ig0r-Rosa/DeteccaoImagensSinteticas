"""Imagem → JSON (BLIP — estável no transformers atual)."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image


MODELO_ID = "Salesforce/blip-image-captioning-large"


class Captionador:
    """Gera descrição estruturada a partir da foto real."""

    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        from transformers import BlipForConditionalGeneration, BlipProcessor

        print(f"[caption] carregando {MODELO_ID} em {self.device}...")
        self.processor = BlipProcessor.from_pretrained(MODELO_ID)
        self.model = BlipForConditionalGeneration.from_pretrained(
            MODELO_ID,
            torch_dtype=self.dtype,
        ).to(self.device)
        self.model.eval()

    def _caption(self, imagem: Image.Image, prompt: str | None = None) -> str:
        if prompt:
            entradas = self.processor(imagem, prompt, return_tensors="pt")
        else:
            entradas = self.processor(imagem, return_tensors="pt")
        entradas = {k: v.to(self.device) for k, v in entradas.items()}
        if "pixel_values" in entradas:
            entradas["pixel_values"] = entradas["pixel_values"].to(self.dtype)
        with torch.inference_mode():
            ids = self.model.generate(**entradas, max_new_tokens=64)
        return self.processor.decode(ids[0], skip_special_tokens=True).strip()

    def descrever(self, caminho: Path) -> dict:
        imagem = Image.open(caminho).convert("RGB")
        # Reduz se for enorme (COCO pode ser grande) — BLIP ok até ~384 internamente
        max_lado = 1024
        if max(imagem.size) > max_lado:
            imagem.thumbnail((max_lado, max_lado), Image.Resampling.LANCZOS)

        cena = self._caption(imagem)
        # Remove prefixo "a photography of" se vier do conditional
        for prefixo in ("a photography of ", "a photo of ", "an image of "):
            if cena.lower().startswith(prefixo):
                cena = cena[len(prefixo) :]
                break

        detalhes = self._caption(imagem, "a photography of")
        for prefixo in ("a photography of ", "a photo of "):
            if detalhes.lower().startswith(prefixo):
                detalhes = detalhes[len(prefixo) :]
                break

        texto = detalhes if len(detalhes) > len(cena) else cena
        prompt = (
            "photorealistic phone photo, natural lighting, "
            f"{texto}, high detail, realistic textures, candid shot"
        )
        return {
            "arquivo": caminho.name,
            "cena": texto,
            "estilo": "foto realista de celular",
            "prompt": prompt,
            "negativo": (
                "cartoon, anime, painting, illustration, 3d render, cgi, "
                "oversaturated, watermark, text, blurry, low quality, deformed"
            ),
            "modelo_caption": MODELO_ID,
        }


def salvar_json(dados: dict, destino: Path) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(dados, ensure_ascii=False, indent=2), encoding="utf-8")
