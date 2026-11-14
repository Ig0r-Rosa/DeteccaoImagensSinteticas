"""Orquestra REAL → JSON → IA com retomada automática.

Uso:
  ./run_pares.sh --limite 5
  ./run_pares.sh
"""

from __future__ import annotations

import argparse
import gc
import traceback
from pathlib import Path

import torch

from .progresso import (
    caption_pronta,
    carregar_estado,
    ia_pronta,
    listar_reais,
    marcar,
    pastas,
    resumo,
    salvar_estado,
)


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pares REAL→JSON→IA (retomável)")
    p.add_argument("--limite", type=int, default=0, help="Máx. imagens (0=todas)")
    p.add_argument("--offset", type=int, default=0, help="Pula as N primeiras")
    p.add_argument("--so-caption", action="store_true", help="Só etapa JSON")
    p.add_argument("--so-gerar", action="store_true", help="Só etapa IA")
    p.add_argument("--forcar", action="store_true", help="Refaz mesmo se existir")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _liberar() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> int:
    args = _parse()
    dirs = pastas()
    for chave in ("json", "ia", "preview"):
        dirs[chave].mkdir(parents=True, exist_ok=True)

    reais = listar_reais(dirs["real"])
    if args.offset:
        reais = reais[args.offset :]
    if args.limite > 0:
        reais = reais[: args.limite]
    if not reais:
        print("Nenhuma imagem em", dirs["real"])
        return 1

    estado = carregar_estado(dirs["estado"])
    n_pool = len(listar_reais(dirs["real"]))
    print("[pares]", resumo(estado, n_pool))
    print(f"[pares] fila desta execução: {len(reais)} | resume=on")

    fazer_cap = not args.so_gerar
    fazer_gen = not args.so_caption
    ok = erro = pulou = 0

    # Fase 1 — captions (modelo leve; depois libera VRAM)
    if fazer_cap:
        pendentes = [
            r
            for r in reais
            if args.forcar or not caption_pronta(dirs["json"], r.stem)
        ]
        print(f"\n=== CAPTION ({len(pendentes)} pendentes) ===")
        if pendentes:
            from .caption import Captionador, salvar_json

            captionador = Captionador()
            for i, real in enumerate(pendentes, 1):
                nome = real.name
                print(f"[caption {i}/{len(pendentes)}] {nome}")
                try:
                    dados = captionador.descrever(real)
                    salvar_json(dados, dirs["json"] / f"{real.stem}.json")
                    marcar(estado, nome, "caption", "ok")
                    ok += 1
                    print("  ok")
                except Exception as exc:
                    erro += 1
                    marcar(estado, nome, "erro", "falhou", str(exc))
                    print("  ERRO:", exc)
                    traceback.print_exc()
                salvar_estado(dirs["estado"], estado)
            del captionador
            _liberar()
        else:
            print("nada a fazer (tudo em cache)")
            for r in reais:
                marcar(estado, r.name, "caption", "ok")
                pulou += 1
            salvar_estado(dirs["estado"], estado)

    # Fase 2 — geração IA
    if fazer_gen:
        pendentes = [
            r
            for r in reais
            if args.forcar or not ia_pronta(dirs["ia"], r.stem)
        ]
        print(f"\n=== GERAR IA ({len(pendentes)} pendentes) ===")
        if pendentes:
            from .gerar import GeradorIA, carregar_json, salvar_ia

            gerador = GeradorIA()
            for i, real in enumerate(pendentes, 1):
                nome = real.name
                json_path = dirs["json"] / f"{real.stem}.json"
                print(f"[gerar {i}/{len(pendentes)}] {nome}")
                try:
                    if not json_path.exists():
                        raise FileNotFoundError(f"Falta JSON: {json_path.name}")
                    dados = carregar_json(json_path)
                    img = gerador.gerar(real, dados, seed=args.seed + i)
                    ia_path = dirs["ia"] / f"{real.stem}.jpg"
                    salvar_ia(img, ia_path)
                    if i <= 8:
                        _salvar_preview(
                            real, ia_path, dirs["preview"] / f"{real.stem}_par.jpg"
                        )
                    marcar(estado, nome, "gerar", "ok")
                    ok += 1
                    print("  ok →", ia_path.name)
                except Exception as exc:
                    erro += 1
                    marcar(estado, nome, "erro", "falhou", str(exc))
                    print("  ERRO:", exc)
                    traceback.print_exc()
                salvar_estado(dirs["estado"], estado)
                _liberar()
            del gerador
            _liberar()
        else:
            print("nada a fazer (tudo em cache)")
            for r in reais:
                marcar(estado, r.name, "gerar", "ok")
                pulou += 1
            salvar_estado(dirs["estado"], estado)

    print("\n=== fim ===")
    print(resumo(estado, n_pool))
    print(f"desta run: ok={ok} pulou={pulou} erro={erro}")
    print("previews:", dirs["preview"])
    return 0 if erro == 0 else 2


def _salvar_preview(real: Path, ia: Path, destino: Path) -> None:
    from PIL import Image

    a = Image.open(real).convert("RGB").resize((512, 512))
    b = Image.open(ia).convert("RGB").resize((512, 512))
    canvas = Image.new("RGB", (1024, 512))
    canvas.paste(a, (0, 0))
    canvas.paste(b, (512, 0))
    destino.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destino, quality=90)
    print("  preview:", destino.name)


if __name__ == "__main__":
    raise SystemExit(main())
