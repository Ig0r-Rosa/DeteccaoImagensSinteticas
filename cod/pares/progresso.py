"""Estado do batch: retoma pulando o que já está pronto."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

EXT_IMG = {".jpg", ".jpeg", ".png", ".webp"}


def raiz_projeto() -> Path:
    return Path(__file__).resolve().parents[2]


def pastas(base: Path | None = None) -> dict[str, Path]:
    root = raiz_projeto()
    base = base or root / "base"
    return {
        "base": base,
        "real": base / "landscape_real",
        "json": base / "pares_json",
        "ia": base / "pares_ia",
        "estado": base / "pares_estado.json",
        "preview": base / "pares_preview",
    }


def listar_reais(pasta_real: Path) -> list[Path]:
    arquivos = [
        p
        for p in sorted(pasta_real.iterdir())
        if p.is_file() and p.suffix.lower() in EXT_IMG and not p.name.lower().startswith("readme")
    ]
    return arquivos


def carregar_estado(caminho: Path) -> dict:
    if not caminho.exists():
        return {"version": 1, "itens": {}, "atualizado": None}
    return json.loads(caminho.read_text(encoding="utf-8"))


def salvar_estado(caminho: Path, estado: dict) -> None:
    estado["atualizado"] = datetime.now(timezone.utc).isoformat()
    caminho.parent.mkdir(parents=True, exist_ok=True)
    tmp = caminho.with_suffix(".tmp")
    tmp.write_text(json.dumps(estado, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(caminho)


def marcar(estado: dict, nome: str, etapa: str, status: str, detalhe: str = "") -> None:
    item = estado.setdefault("itens", {}).setdefault(nome, {})
    item[etapa] = status
    if detalhe:
        item["detalhe"] = detalhe[:500]
    elif "detalhe" in item and status == "ok":
        item.pop("detalhe", None)


def caption_pronta(pasta_json: Path, stem: str) -> bool:
    return (pasta_json / f"{stem}.json").exists()


def ia_pronta(pasta_ia: Path, stem: str, sufixo: str = ".jpg") -> bool:
    return (pasta_ia / f"{stem}{sufixo}").exists()


def resumo(estado: dict, n_total: int) -> str:
    itens = estado.get("itens", {})
    cap_ok = sum(1 for v in itens.values() if v.get("caption") == "ok")
    gen_ok = sum(1 for v in itens.values() if v.get("gerar") == "ok")
    erros = sum(1 for v in itens.values() if "erro" in v.values())
    return f"total={n_total} caption_ok={cap_ok} gerar_ok={gen_ok} erros={erros}"
