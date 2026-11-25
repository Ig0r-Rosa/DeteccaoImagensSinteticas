"""Dados: pares REAL × FAKE, letterbox e splits."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import cv2
import numpy as np

EXTENSOES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
FAKE, REAL = 0, 1
FEATURE_SIZE = 224  # entrada do SVM (imagem inteira)
HOG_SIZE = 128      # HOG usa resolução menor (mais rápido)


def _indice_por_stem(pasta: Path) -> dict[str, Path]:
    """Mapeia nome do arquivo (sem extensão) → caminho."""
    mapa: dict[str, Path] = {}
    if not pasta.exists():
        return mapa
    for arquivo in sorted(pasta.iterdir()):
        if not arquivo.is_file() or arquivo.suffix.lower() not in EXTENSOES:
            continue
        if arquivo.name.lower().startswith("readme"):
            continue
        mapa.setdefault(arquivo.stem, arquivo)
    return mapa


def _imread_seguro(caminho: str | Path) -> np.ndarray | None:
    """Lê imagem (OpenCV; fallback PIL se o path tiver unicode)."""
    caminho = Path(caminho)
    try:
        dados = np.fromfile(str(caminho), dtype=np.uint8)
        if dados.size > 0:
            imagem = cv2.imdecode(dados, cv2.IMREAD_COLOR)
            if imagem is not None:
                return imagem
    except OSError:
        pass
    try:
        from PIL import Image

        with Image.open(caminho) as pil:
            rgb = np.asarray(pil.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def letterbox_rgb(imagem: np.ndarray, size: int = FEATURE_SIZE) -> np.ndarray:
    """Encaixa a imagem inteira em size×size (borda espelhada)."""
    if imagem.ndim == 2:
        imagem = cv2.cvtColor(imagem, cv2.COLOR_GRAY2RGB)
    altura, largura = imagem.shape[:2]
    if altura == size and largura == size:
        return imagem
    escala = size / float(max(altura, largura))
    nova_h = max(1, int(round(altura * escala)))
    nova_w = max(1, int(round(largura * escala)))
    redim = cv2.resize(imagem, (nova_w, nova_h), interpolation=cv2.INTER_AREA)
    top = (size - nova_h) // 2
    bottom = size - nova_h - top
    left = (size - nova_w) // 2
    right = size - nova_w - left
    return cv2.copyMakeBorder(
        redim, top, bottom, left, right, borderType=cv2.BORDER_REFLECT_101
    )


def carregar_imagem(caminho: str, size: int = FEATURE_SIZE) -> np.ndarray:
    """Carrega RGB e aplica letterbox."""
    imagem = _imread_seguro(caminho)
    if imagem is None:
        raise ValueError(f"Não foi possível carregar: {caminho}")
    return letterbox_rgb(cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB), size)


def resumo_rotulos(rotulos: list[int]) -> str:
    """Contagem por classe para log."""
    return str(dict(Counter(rotulos)))


def avisar_qualidade_bases(meta: dict[str, int]) -> list[str]:
    """Avisos se houver poucos pares."""
    avisos: list[str] = []
    if meta.get("pares", 0) == 0:
        avisos.append(
            "Nenhum par encontrado. Precisa de base/landscape_real + base/pares_ia."
        )
    elif meta.get("pares", 0) < 1000:
        avisos.append(f"Poucos pares ({meta['pares']}). Ideal ~5k.")
    return avisos


def carregar_datasets_pares(
    base_dir: Path,
    seed: int = 42,
    n_treino: int = 4000,
    n_teste: int = 1000,
) -> tuple[list[str], list[int], list[str], list[int], dict[str, int]]:
    """Carrega pares alinhados (mesmo stem em landscape_real e pares_ia).

    n_teste=0 → só treino (avaliação no app).
    """
    reais = _indice_por_stem(base_dir / "landscape_real")
    fakes = _indice_por_stem(base_dir / "pares_ia")
    stems = sorted(set(reais) & set(fakes))
    if not stems:
        raise ValueError(
            "Nenhum par REAL×IA. Confira base/landscape_real e base/pares_ia."
        )

    rng = np.random.default_rng(seed)
    ordem = list(stems)
    rng.shuffle(ordem)

    n_teste = max(0, int(n_teste))
    n_treino = max(0, int(n_treino))
    if n_teste == 0:
        n_tr = min(n_treino, len(ordem)) if n_treino else len(ordem)
        stems_tr, stems_te = ordem[:n_tr], []
    else:
        n_te = min(n_teste, len(ordem) - 1) if len(ordem) > 1 else 0
        n_tr = min(n_treino, len(ordem) - n_te)
        stems_te = ordem[:n_te]
        stems_tr = ordem[n_te : n_te + n_tr]

    def _montar(stems_sel: list[str]) -> tuple[list[str], list[int]]:
        caminhos, rotulos = [], []
        for stem in stems_sel:
            caminhos.append(str(fakes[stem]))
            rotulos.append(FAKE)
            caminhos.append(str(reais[stem]))
            rotulos.append(REAL)
        return caminhos, rotulos

    train_c, train_y = _montar(stems_tr)
    test_c, test_y = _montar(stems_te)

    if train_c:
        perm = rng.permutation(len(train_c))
        train_c = [train_c[i] for i in perm]
        train_y = [train_y[i] for i in perm]
    if test_c:
        perm = rng.permutation(len(test_c))
        test_c = [test_c[i] for i in perm]
        test_y = [test_y[i] for i in perm]

    meta = {
        "pares": len(stems),
        "pares_treino": len(stems_tr),
        "pares_teste": len(stems_te),
        "landscape_real": len(reais),
        "pares_ia": len(fakes),
        "train_total": len(train_c),
        "test_total": len(test_c),
        "feature_size": FEATURE_SIZE,
    }
    return train_c, train_y, test_c, test_y, meta
