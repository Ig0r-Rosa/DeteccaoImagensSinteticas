"""JSON + foto REAL → imagem IA (SD1.5 + ControlNet Depth)."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image


CONTROLNET_ID = "lllyasviel/sd-controlnet-depth"
SD_ID = "sd-legacy/stable-diffusion-v1-5"
SIZE = 512  # cabe na RTX 3050 6GB


class GeradorIA:
    """Mantém a geometria da cena (depth) e gera aparência sintética."""

    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        from controlnet_aux import MidasDetector
        from diffusers import (
            ControlNetModel,
            StableDiffusionControlNetPipeline,
            UniPCMultistepScheduler,
        )

        print(f"[gerar] carregando ControlNet + SD1.5 em {self.device}...")
        self.depth = MidasDetector.from_pretrained("lllyasviel/Annotators")
        controlnet = ControlNetModel.from_pretrained(CONTROLNET_ID, torch_dtype=self.dtype)
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            SD_ID,
            controlnet=controlnet,
            torch_dtype=self.dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_attention_slicing()
        if self.device == "cuda":
            try:
                self.pipe.enable_model_cpu_offload()
            except Exception:
                self.pipe = self.pipe.to(self.device)
        else:
            self.pipe = self.pipe.to(self.device)

    def _quadrado(self, imagem: Image.Image, lado: int = SIZE) -> Image.Image:
        return imagem.convert("RGB").resize((lado, lado), Image.Resampling.LANCZOS)

    def gerar(
        self,
        caminho_real: Path,
        dados_json: dict,
        seed: int = 42,
        steps: int = 20,
        guidance: float = 7.0,
        control_scale: float = 0.9,
    ) -> Image.Image:
        real = self._quadrado(Image.open(caminho_real))
        mapa = self.depth(real)
        if not isinstance(mapa, Image.Image):
            mapa = Image.fromarray(mapa)
        mapa = self._quadrado(mapa)

        gerador = torch.Generator(device="cpu").manual_seed(seed)
        resultado = self.pipe(
            prompt=dados_json.get("prompt") or dados_json.get("cena", "a photo"),
            negative_prompt=dados_json.get("negativo", ""),
            image=mapa,
            num_inference_steps=steps,
            guidance_scale=guidance,
            controlnet_conditioning_scale=control_scale,
            generator=gerador,
        )
        return resultado.images[0]


def carregar_json(caminho: Path) -> dict:
    return json.loads(caminho.read_text(encoding="utf-8"))


def salvar_ia(imagem: Image.Image, destino: Path) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    imagem.save(destino, quality=92)
