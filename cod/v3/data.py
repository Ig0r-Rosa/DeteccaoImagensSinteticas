"""Dados v3: carrega FAKE (instagram_ai) + REAL (landscape_real); letterbox; splits."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import cv2
import numpy as np

EXTENSOES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
FAKE, REAL = 0, 1
# Tamanho fixo: imagem inteira com proporção (letterbox).
FEATURE_SIZE = 224
# HOG (e a parte HOG da fusão) usa resolução menor — bem mais rápido no treino.
HOG_SIZE = 128


def listar_pasta_plana(pasta: Path, rotulo: int) -> tuple[list[str], list[int]]:
    """Lista imagens em pasta plana ou com subpastas."""
    caminhos, rotulos = [], []
    if not pasta.exists():
        return caminhos, rotulos
    for arquivo in sorted(pasta.rglob("*")):
        if arquivo.is_file() and arquivo.suffix.lower() in EXTENSOES:
            if arquivo.name.lower().startswith("readme"):
                continue
            caminhos.append(str(arquivo))
            rotulos.append(rotulo)
    return caminhos, rotulos


def juntar_fontes(*pares: tuple[list[str], list[int]]) -> tuple[list[str], list[int]]:
    caminhos, rotulos = [], []
    for c, r in pares:
        caminhos.extend(c)
        rotulos.extend(r)
    return caminhos, rotulos


def split_holdout(
    caminhos: list[str],
    rotulos: list[int],
    frac_teste: float = 0.2,
    seed: int = 42,
) -> tuple[list[str], list[int], list[str], list[int]]:
    """Separa holdout estratificado por classe."""
    rng = np.random.default_rng(seed)
    train_c, train_y, test_c, test_y = [], [], [], []
    for classe in (FAKE, REAL):
        idx = [i for i, r in enumerate(rotulos) if r == classe]
        if not idx:
            continue
        rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * frac_teste))) if len(idx) > 4 else 0
        test_idx = set(idx[:n_test])
        for i in idx:
            if i in test_idx:
                test_c.append(caminhos[i])
                test_y.append(rotulos[i])
            else:
                train_c.append(caminhos[i])
                train_y.append(rotulos[i])
    return train_c, train_y, test_c, test_y


def _amostrar_indices(idx: list[int], n: int, rng: np.random.Generator) -> list[int]:
    if n <= 0 or not idx:
        return []
    n = min(n, len(idx))
    return list(rng.choice(idx, size=n, replace=False))


def criar_subconjunto(
    caminhos: list[str],
    rotulos: list[int],
    quantidade_por_classe: int,
    seed: int = 42,
) -> tuple[list[str], list[int]]:
    """Amostra balanceada por classe (Instagram AI × fotos reais)."""
    rng = np.random.default_rng(seed)
    escolhidos: list[int] = []
    for classe in (FAKE, REAL):
        idx_classe = [i for i, r in enumerate(rotulos) if r == classe]
        n_alvo = min(quantidade_por_classe, len(idx_classe))
        escolhidos.extend(_amostrar_indices(idx_classe, n_alvo, rng))
    if not escolhidos:
        raise ValueError(
            "Subconjunto vazio. Confira base/instagram_ai e base/landscape_real."
        )
    rng.shuffle(escolhidos)
    return [caminhos[i] for i in escolhidos], [rotulos[i] for i in escolhidos]


def _imread_seguro(caminho: str | Path) -> np.ndarray | None:
    """Lê imagem com espaços/unicode; tenta OpenCV e depois PIL."""
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
    """Encaixa a imagem INTEIRA em size×size (proporção).

    Usa borda espelhada — faixas pretas dominam a FFT e matam a precisão.
    """
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
    """Carrega RGB e prepara a imagem inteira (letterbox) para o SVM."""
    imagem = _imread_seguro(caminho)
    if imagem is None:
        raise ValueError(f"Não foi possível carregar: {caminho}")
    imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
    return letterbox_rgb(imagem, size)


def filtrar_imagens_legiveis(
    caminhos: list[str], rotulos: list[int]
) -> tuple[list[str], list[int], int]:
    """Remove arquivos ilegíveis."""
    ok_c, ok_y, ruins = [], [], 0
    for caminho, rotulo in zip(caminhos, rotulos):
        if _imread_seguro(caminho) is None:
            ruins += 1
            continue
        ok_c.append(caminho)
        ok_y.append(rotulo)
    return ok_c, ok_y, ruins


def resumo_rotulos(rotulos: list[int]) -> str:
    return str(dict(Counter(rotulos)))


def avisar_qualidade_bases(meta: dict[str, int]) -> list[str]:
    """Alertas: FAKE Instagram + REAL diversa (ex. COCO) em landscape_real."""
    avisos: list[str] = []
    if meta.get("instagram_ai", 0) == 0:
        avisos.append("base/instagram_ai vazia — falta FAKE de rede social.")
    if meta.get("landscape_real", 0) == 0:
        avisos.append(
            "base/landscape_real vazia — coloque fotos REAIS (ex. COCO 2017). "
            "Ver base/README.md."
        )
    elif meta.get("landscape_real", 0) < 5000:
        avisos.append(
            f"Poucas REAL ({meta['landscape_real']}). "
            "Ideal ~mesmo volume do Instagram AI (~50–60k)."
        )
    n_fake, n_real = meta.get("instagram_ai", 0), meta.get("landscape_real", 0)
    if n_fake > 0 and n_real > 0 and n_real < 0.4 * n_fake:
        avisos.append(
            f"Desbalanceado: FAKE={n_fake} REAL={n_real}. "
            "Aumente landscape_real para ficar perto do número de FAKE."
        )
    return avisos


def carregar_datasets_mistos(
    base_dir: Path,
    seed: int = 42,
    frac_holdout: float = 0.15,
) -> tuple[list[str], list[int], list[str], list[int], dict[str, int]]:
    """Treino/teste: Instagram AI (FAKE) + landscape_real (REAL). Sem CelebA/CIFAKE."""
    insta = listar_pasta_plana(base_dir / "instagram_ai", FAKE)
    reais = listar_pasta_plana(base_dir / "landscape_real", REAL)
    todos = juntar_fontes(insta, reais)

    if len(todos[0]) == 0:
        raise ValueError(
            "Nenhuma imagem encontrada. "
            "Preencha base/instagram_ai e base/landscape_real."
        )

    if frac_holdout and frac_holdout > 0:
        train_c, train_y, test_c, test_y = split_holdout(
            *todos, frac_teste=frac_holdout, seed=seed
        )
    else:
        train_c, train_y = todos
        test_c, test_y = [], []

    meta = {
        "cifake_train": 0,
        "cifake_test": 0,
        "instagram_ai": len(insta[0]),
        "celeba_real": 0,
        "landscape_real": len(reais[0]),
        "externo_train": len(train_c),
        "externo_test": len(test_c),
        "train_total": len(train_c),
        "test_total": len(test_c),
        "feature_size": FEATURE_SIZE,
    }
    return train_c, train_y, test_c, test_y, meta
