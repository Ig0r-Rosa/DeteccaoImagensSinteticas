"""Features v3: FFT / HOG / LBP, augs de treino e transformações de robustez."""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from tqdm import tqdm

from .data import FEATURE_SIZE, HOG_SIZE, carregar_imagem, letterbox_rgb

SEED = 42


def extrair_features_fft(imagem: np.ndarray, num_bins: int = 64) -> np.ndarray:
    """Perfil radial + energia em faixas (sem min-max por imagem).

    Min-max antigo apagava diferença de energia HF (útil p/ IA vs real).
    StandardScaler no treino normaliza entre amostras.
    """
    cinza = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY) if imagem.ndim == 3 else imagem
    espectro = np.log(np.abs(np.fft.fftshift(np.fft.fft2(cinza.astype(np.float64)))) + 1e-8)
    h, w = espectro.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    raios = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r_max = float(np.sqrt(cx**2 + cy**2))
    bins = np.linspace(0, r_max, num_bins + 1)
    radial = np.zeros(num_bins, dtype=np.float64)
    for i in range(num_bins):
        mask = (raios >= bins[i]) & (raios < bins[i + 1])
        if np.any(mask):
            radial[i] = espectro[mask].mean()

    # Faixas de frequência: baixa / média / alta (sinais típicos de síntese).
    r_norm = raios / max(r_max, 1e-8)
    e_tot = float(espectro.sum()) + 1e-8
    e_low = float(espectro[r_norm < 0.25].sum()) / e_tot
    e_mid = float(espectro[(r_norm >= 0.25) & (r_norm < 0.60)].sum()) / e_tot
    e_high = float(espectro[r_norm >= 0.60].sum()) / e_tot
    pesos = radial - radial.min()
    pesos = pesos / (pesos.sum() + 1e-8)
    centroide = float(np.dot(pesos, np.linspace(0, 1, num_bins)))
    return np.concatenate([radial, [e_low, e_mid, e_high, centroide]])


def _pixels_por_celula(lado: int) -> tuple[int, int]:
    """Célula HOG: em 128 usa 16×16 (~1568 dims); em 64 usa 8×8."""
    if lado >= 128:
        return (16, 16)
    return (8, 8)


def extrair_features_hog(imagem: np.ndarray) -> np.ndarray:
    """HOG só na versão reduzida (HOG_SIZE) — FFT/LBP continuam em FEATURE_SIZE."""
    if imagem.shape[0] != HOG_SIZE or imagem.shape[1] != HOG_SIZE:
        imagem = letterbox_rgb(imagem, HOG_SIZE)
    cinza = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY) if imagem.ndim == 3 else imagem
    cell = _pixels_por_celula(HOG_SIZE)
    return hog(
        cinza,
        orientations=8,
        pixels_per_cell=cell,
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=False,
    )


def extrair_features_lbp(imagem: np.ndarray, P: int = 8, R: int = 1) -> np.ndarray:
    cinza = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY) if imagem.ndim == 3 else imagem
    mapa = local_binary_pattern(cinza, P=P, R=R, method="uniform")
    hist, _ = np.histogram(mapa.ravel(), bins=P + 2, range=(0, P + 2))
    hist = hist.astype(np.float64)
    if hist.sum() > 0:
        hist /= hist.sum()
    return hist


EXTRATORES = {
    "fft": extrair_features_fft,
    "hog": extrair_features_hog,
    "lbp": extrair_features_lbp,
}


def jpeg_compress(imagem: np.ndarray, qualidade: int = 70) -> np.ndarray:
    bgr = cv2.cvtColor(imagem, cv2.COLOR_RGB2BGR)
    ok, buffer = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(qualidade)])
    if not ok:
        raise RuntimeError("Falha ao comprimir JPEG.")
    return cv2.cvtColor(cv2.imdecode(buffer, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def crop_aleatorio(imagem: np.ndarray, percentual: float = 0.25, seed: int | None = None) -> np.ndarray:
    h, w = imagem.shape[:2]
    novo_h = max(1, int(h * (1 - percentual)))
    novo_w = max(1, int(w * (1 - percentual)))
    rng = np.random.default_rng(seed)
    y0 = int(rng.integers(0, h - novo_h + 1))
    x0 = int(rng.integers(0, w - novo_w + 1))
    recorte = imagem[y0 : y0 + novo_h, x0 : x0 + novo_w]
    return cv2.resize(recorte, (w, h), interpolation=cv2.INTER_LINEAR)


def resize_down_up(imagem: np.ndarray, escala: float = 0.5) -> np.ndarray:
    h, w = imagem.shape[:2]
    reduzida = cv2.resize(
        imagem, (max(1, int(w * escala)), max(1, int(h * escala))), interpolation=cv2.INTER_AREA
    )
    return cv2.resize(reduzida, (w, h), interpolation=cv2.INTER_LINEAR)


def blur(imagem: np.ndarray, ksize: int = 3) -> np.ndarray:
    k = int(ksize) | 1
    return cv2.GaussianBlur(imagem, (k, k), 0)


def pipeline_compartilhamento(imagem: np.ndarray, seed: int = SEED) -> np.ndarray:
    img = crop_aleatorio(imagem, 0.25, seed=seed)
    img = resize_down_up(img, 0.5)
    return jpeg_compress(img, 70)


TRANSFORMACOES = {
    "jpeg_90": lambda img, seed=None: jpeg_compress(img, 90),
    "jpeg_70": lambda img, seed=None: jpeg_compress(img, 70),
    "jpeg_50": lambda img, seed=None: jpeg_compress(img, 50),
    "crop_10": lambda img, seed=None: crop_aleatorio(img, 0.10, seed=SEED if seed is None else seed),
    "crop_25": lambda img, seed=None: crop_aleatorio(img, 0.25, seed=SEED if seed is None else seed),
    "crop_40": lambda img, seed=None: crop_aleatorio(img, 0.40, seed=SEED if seed is None else seed),
    "resize_0.5": lambda img, seed=None: resize_down_up(img, 0.5),
    "blur_3": lambda img, seed=None: blur(img, 3),
    "pipeline": lambda img, seed=None: pipeline_compartilhamento(img, seed=SEED if seed is None else seed),
}
# Treino: augs leves. Crops fortes + jpeg_50 destroem o sinal espectral da FFT.
VARIANTES_TREINO = [
    "original",
    "jpeg_90",
    "jpeg_70",
    "resize_0.5",
    "blur_3",
    "pipeline",
]
# Avaliação de robustez no notebook continua com todas.
VARIANTES_TESTE = ["original"] + list(TRANSFORMACOES.keys())


def seed_variante(caminho: str, nome: str) -> int:
    chave = f"{caminho}|{nome}|{SEED}".encode()
    return int(hashlib.md5(chave).hexdigest()[:8], 16) % (2**31 - 1)


def aplicar_variante(imagem: np.ndarray, nome: str, seed: int | None = None) -> np.ndarray:
    if nome == "original":
        return imagem
    return TRANSFORMACOES[nome](imagem, seed=seed)


def carregar_npz(caminho: Path):
    if not caminho.exists():
        return None
    try:
        return np.load(caminho)
    except Exception as erro:
        print(f"Cache ilegível ({caminho.name}): {erro}")
        try:
            caminho.unlink()
        except OSError:
            pass
        return None


def extrair_treino_aumentado(
    caminhos: list[str],
    rotulos: list[int],
    tipo_feature: str,
    pasta_saida: Path,
    tag: str = "v3",
    forcar: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    pasta_saida.mkdir(parents=True, exist_ok=True)
    n, n_var = len(caminhos), len(VARIANTES_TREINO)
    caminho = pasta_saida / f"features_{tag}_{tipo_feature}_train_aug_n{n}_v{n_var}.npz"
    if not forcar:
        dados = carregar_npz(caminho)
        if dados is not None:
            print(f"Cache aug: {caminho.name} | X={dados['X'].shape}")
            return dados["X"], dados["y"]
    funcao = EXTRATORES[tipo_feature]
    vetores, ys = [], []
    puladas = 0
    for caminho_img, y in tqdm(zip(caminhos, rotulos), total=n, desc=f"aug-{tipo_feature}"):
        try:
            img = carregar_imagem(caminho_img, FEATURE_SIZE)
        except ValueError:
            puladas += 1
            continue
        for nome in VARIANTES_TREINO:
            img_t = aplicar_variante(img, nome, seed=seed_variante(caminho_img, nome))
            vetores.append(funcao(img_t))
            ys.append(y)
    if puladas:
        print(f"Avisos: {puladas} imagens ilegíveis puladas em {tipo_feature}.")
    if not vetores:
        raise ValueError(f"Nenhuma imagem válida para {tipo_feature}.")
    X, y_out = np.asarray(vetores), np.asarray(ys)
    np.savez_compressed(caminho, X=X, y=y_out)
    print(f"Salvo aug: {caminho.name} | X={X.shape}")
    return X, y_out


def extrair_teste(
    caminhos: list[str],
    rotulos: list[int],
    tipo_feature: str,
    pasta_saida: Path,
    tag: str = "v3",
    forcar: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    pasta_saida.mkdir(parents=True, exist_ok=True)
    caminho = pasta_saida / f"features_{tag}_{tipo_feature}_test_n{len(caminhos)}.npz"
    if not forcar:
        dados = carregar_npz(caminho)
        if dados is not None:
            print(f"Cache teste: {caminho.name} | X={dados['X'].shape}")
            return dados["X"], dados["y"]
    funcao = EXTRATORES[tipo_feature]
    vetores, ys, puladas = [], [], 0
    for c, y in tqdm(zip(caminhos, rotulos), total=len(caminhos), desc=tipo_feature):
        try:
            vetores.append(funcao(carregar_imagem(c, FEATURE_SIZE)))
            ys.append(y)
        except ValueError:
            puladas += 1
    if puladas:
        print(f"Avisos: {puladas} imagens ilegíveis puladas no teste ({tipo_feature}).")
    if not vetores:
        raise ValueError(f"Nenhuma imagem válida no teste para {tipo_feature}.")
    X, y = np.asarray(vetores), np.asarray(ys)
    np.savez_compressed(caminho, X=X, y=y)
    print(f"Salvo teste: {caminho.name} | X={X.shape}")
    return X, y


def extrair_teste_transformado(
    caminhos: list[str],
    rotulos: list[int],
    tipo_feature: str,
    nome_transf: str,
    pasta_saida: Path,
    tag: str = "v3",
    forcar: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    pasta_saida.mkdir(parents=True, exist_ok=True)
    caminho = pasta_saida / f"features_{tag}_{tipo_feature}_test_{nome_transf}_n{len(caminhos)}.npz"
    if not forcar:
        dados = carregar_npz(caminho)
        if dados is not None:
            return dados["X"], dados["y"]
    funcao = EXTRATORES[tipo_feature]
    vetores, ys = [], []
    for c, rotulo in tqdm(zip(caminhos, rotulos), total=len(caminhos), desc=f"{tipo_feature}-{nome_transf}"):
        try:
            img = carregar_imagem(c, FEATURE_SIZE)
        except ValueError:
            continue
        img_t = aplicar_variante(img, nome_transf, seed=SEED)
        vetores.append(funcao(img_t))
        ys.append(rotulo)
    X, y = np.asarray(vetores), np.asarray(ys)
    np.savez_compressed(caminho, X=X, y=y)
    return X, y
