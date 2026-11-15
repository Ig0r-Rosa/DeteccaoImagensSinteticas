"""Dados v3: pares REAL (landscape_real) × FAKE (pares_ia); letterbox; splits."""

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


def _indice_por_stem(pasta: Path) -> dict[str, Path]:
    """Mapeia stem → caminho (primeira extensão encontrada)."""
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


def listar_stems_pareados(base_dir: Path) -> list[str]:
    """Stems presentes em landscape_real e pares_ia (mesmo nome = par)."""
    reais = _indice_por_stem(base_dir / "landscape_real")
    fakes = _indice_por_stem(base_dir / "pares_ia")
    return sorted(set(reais) & set(fakes))


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
    """Amostra balanceada por classe."""
    if quantidade_por_classe <= 0 or not caminhos:
        return [], []
    rng = np.random.default_rng(seed)
    escolhidos: list[int] = []
    for classe in (FAKE, REAL):
        idx_classe = [i for i, r in enumerate(rotulos) if r == classe]
        n_alvo = min(quantidade_por_classe, len(idx_classe))
        escolhidos.extend(_amostrar_indices(idx_classe, n_alvo, rng))
    if not escolhidos:
        raise ValueError(
            "Subconjunto vazio. Confira base/landscape_real e base/pares_ia."
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
    """Alertas para pares REAL × IA."""
    avisos: list[str] = []
    if meta.get("pares", 0) == 0:
        avisos.append(
            "Nenhum par encontrado. Precisa de base/landscape_real + base/pares_ia "
            "com o mesmo nome de arquivo."
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
    """Carrega pares alinhados: REAL=landscape_real, FAKE=pares_ia (mesmo stem).

    O split é por *par* (stem): o REAL e o FAKE do mesmo stem ficam no mesmo split.
    n_teste=0 → treino com todos os pares; teste vazio (avaliação só no app).
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
        # Modo app: usa até n_treino pares (ou todos se n_treino >= pool).
        n_tr = min(n_treino, len(ordem)) if n_treino else len(ordem)
        stems_tr, stems_te = ordem[:n_tr], []
    else:
        n_te = min(n_teste, len(ordem) - 1) if len(ordem) > 1 else 0
        n_tr = min(n_treino, len(ordem) - n_te)
        stems_te = ordem[:n_te]
        stems_tr = ordem[n_te : n_te + n_tr]

    def _montar(stems_sel: list[str]) -> tuple[list[str], list[int]]:
        c, y = [], []
        for stem in stems_sel:
            c.append(str(fakes[stem]))
            y.append(FAKE)
            c.append(str(reais[stem]))
            y.append(REAL)
        return c, y

    train_c, train_y = _montar(stems_tr)
    test_c, test_y = _montar(stems_te)

    # Embaralha amostras (mantém balanceamento).
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


def carregar_datasets_mistos(
    base_dir: Path,
    seed: int = 42,
    frac_holdout: float = 0.15,
) -> tuple[list[str], list[int], list[str], list[int], dict[str, int]]:
    """Compat: delega aos pares (holdout ~frac por par)."""
    stems = listar_stems_pareados(base_dir)
    n = len(stems)
    n_teste = int(round(n * frac_holdout)) if frac_holdout and n else 0
    n_treino = max(0, n - n_teste)
    return carregar_datasets_pares(
        base_dir, seed=seed, n_treino=n_treino, n_teste=n_teste
    )
