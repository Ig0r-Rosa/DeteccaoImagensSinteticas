"""Pipeline de inferencia por blocos para o classificador treinado no notebook."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import cv2
import joblib
import numpy as np
from PIL import Image
from skimage.feature import hog, local_binary_pattern
from sklearn.exceptions import InconsistentVersionWarning

BLOCK_SIZE = 32
FEATURE_TYPES = ("fft", "hog", "lbp")


def _load_trusted_artifact(path: Path):
    """Carrega somente artefatos locais confiáveis, isolando o aviso de versão conhecido."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
        return joblib.load(path)


@dataclass(frozen=True)
class DetectionResult:
    probability_map: np.ndarray
    mean_ai_probability: float
    simple_mean_ai_probability: float
    high_evidence_fraction: float
    regional_evidence: float
    region_threshold: float
    detected_regions: int
    overlay: Image.Image
    rows: int
    columns: int
    analysis_windows: int
    scales: tuple[int, ...]
    probability_is_calibrated: bool


def extract_fft(image: np.ndarray, num_bins: int = 64) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    spectrum = np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray))) + 1e-8)
    height, width = spectrum.shape
    center_y, center_x = height // 2, width // 2
    y, x = np.ogrid[:height, :width]
    radii = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    max_radius = np.sqrt(center_x**2 + center_y**2)
    bins = np.linspace(0, max_radius, num_bins + 1)
    vector = np.zeros(num_bins, dtype=np.float64)
    for index in range(num_bins):
        mask = (radii >= bins[index]) & (radii < bins[index + 1])
        if np.any(mask):
            vector[index] = spectrum[mask].mean()
    value_range = np.ptp(vector)
    if value_range > 1e-8:
        vector = (vector - vector.min()) / value_range
    return vector


def extract_hog(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return hog(
        gray,
        orientations=8,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=False,
    )


def extract_lbp(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    histogram, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10))
    histogram = histogram.astype(np.float64)
    if histogram.sum() > 0:
        histogram /= histogram.sum()
    return histogram


def split_into_blocks(image: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """Divide a imagem em 32x32, refletindo apenas as bordas incompletas."""
    height, width = image.shape[:2]
    pad_height = (-height) % BLOCK_SIZE
    pad_width = (-width) % BLOCK_SIZE
    border = cv2.BORDER_REFLECT_101 if height > 1 and width > 1 else cv2.BORDER_REPLICATE
    padded = cv2.copyMakeBorder(image, 0, pad_height, 0, pad_width, border)
    rows, columns = padded.shape[0] // BLOCK_SIZE, padded.shape[1] // BLOCK_SIZE
    blocks = (
        padded.reshape(rows, BLOCK_SIZE, columns, BLOCK_SIZE, 3)
        .transpose(0, 2, 1, 3, 4)
        .reshape(-1, BLOCK_SIZE, BLOCK_SIZE, 3)
    )
    return blocks, (rows, columns)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -40, 40)
    return 1.0 / (1.0 + np.exp(-values))


def analyze_evidence_regions(
    probability_map: np.ndarray,
) -> tuple[float, float, int, float, np.ndarray]:
    """Detecta regiões conexas fortes e as combina com a média global.

    Retorna resultado final, limiar adaptativo, quantidade de regiões, força da
    região mais relevante e rótulos espaciais. A confiança de área satura
    gradualmente, evitando que um único ponto vermelho domine a imagem inteira.
    """
    values = np.asarray(probability_map, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("O mapa de probabilidades deve conter valores finitos.")
    simple_mean = float(values.mean())
    threshold = float(min(0.8, max(0.5, simple_mean + 0.5 * values.std())))
    mask = (values >= threshold).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    strongest = 0.0
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        region_mean = float(values[labels == label].mean())
        area_confidence = 1.0 - np.exp(-area / 4.0)
        strongest = max(strongest, region_mean * area_confidence)
    # União de duas evidências: base global e região espacial mais forte.
    combined = 1.0 - (1.0 - simple_mean) * (1.0 - strongest)
    return float(combined), threshold, count - 1, strongest, labels


class SyntheticImageDetector:
    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        robust = (self.model_dir / "svm_robusto_fusion.pkl").exists()
        prefix = "robusto_" if robust else ""
        model_name = f"svm_{prefix}fusion.pkl"
        required = [model_name, *(f"scaler_{prefix}{kind}.pkl" for kind in FEATURE_TYPES)]
        missing = [name for name in required if not (self.model_dir / name).exists()]
        if missing:
            raise FileNotFoundError(
                f"Artefatos ausentes em {self.model_dir}: {', '.join(missing)}"
            )
        # Arquivos joblib devem vir exclusivamente do treinamento confiavel do projeto.
        self.model = _load_trusted_artifact(self.model_dir / model_name)
        self.scalers = {
            kind: _load_trusted_artifact(self.model_dir / f"scaler_{prefix}{kind}.pkl")
            for kind in FEATURE_TYPES
        }

    def _features(self, blocks: np.ndarray) -> np.ndarray:
        extractors = {"fft": extract_fft, "hog": extract_hog, "lbp": extract_lbp}
        scaled = []
        for kind in FEATURE_TYPES:
            values = np.asarray([extractors[kind](block) for block in blocks])
            scaled.append(self.scalers[kind].transform(values))
        return np.hstack(scaled)

    def _ai_probabilities(self, features: np.ndarray) -> tuple[np.ndarray, bool]:
        classes = np.asarray(self.model.classes_)
        fake_positions = np.flatnonzero(classes == 0)
        if len(classes) != 2 or len(fake_positions) != 1:
            raise ValueError(f"Classes incompatíveis; esperado [0, 1], recebido {classes.tolist()}")
        fake_index = int(fake_positions[0])
        if hasattr(self.model, "predict_proba") and getattr(self.model, "probability", True):
            return self.model.predict_proba(features)[:, fake_index], True
        decision = np.asarray(self.model.decision_function(features), dtype=float)
        # Em SVC binario, valores positivos apontam para classes_[1].
        fake_score = decision if fake_index == 1 else -decision
        return _sigmoid(fake_score), False

    def detect(
        self,
        image: Image.Image,
        overlay_alpha: float = 0.5,
        scales: tuple[int, ...] = (32, 64, 128),
    ) -> DetectionResult:
        """Analisa janelas sobrepostas, sempre redimensionadas para a entrada 32x32.

        A sobreposição de 50% cria uma segunda passagem no meio dos vãos entre
        blocos. Janelas maiores dão contexto adicional e cada pixel recebe a
        média de todas as janelas que o cobrem.
        """
        scales = tuple(sorted(set(int(scale) for scale in scales)))
        if not scales or any(scale < BLOCK_SIZE or scale % BLOCK_SIZE for scale in scales):
            raise ValueError("As escalas devem ser múltiplos de 32 e não podem ser vazias.")
        rgb = np.asarray(image.convert("RGB"))
        height, width = rgb.shape[:2]
        probability_sum = np.zeros((height, width), dtype=np.float64)
        coverage = np.zeros((height, width), dtype=np.uint16)
        calibrated = False
        window_count = 0

        for scale in scales:
            windows, regions = _analysis_windows(rgb, scale)
            probabilities, scale_calibrated = self._ai_probabilities(self._features(windows))
            calibrated = scale_calibrated
            window_count += len(windows)
            for probability, (top, left, bottom, right) in zip(probabilities, regions):
                probability_sum[top:bottom, left:right] += probability
                coverage[top:bottom, left:right] += 1

        pixel_map = probability_sum / np.maximum(coverage, 1)
        rows = int(np.ceil(height / BLOCK_SIZE))
        columns = int(np.ceil(width / BLOCK_SIZE))
        probability_map = cv2.resize(
            pixel_map, (columns, rows), interpolation=cv2.INTER_AREA
        )
        final_probability, threshold, region_count, regional_evidence, labels = (
            analyze_evidence_regions(probability_map)
        )
        overlay = create_overlay(rgb, pixel_map, overlay_alpha, labels)
        return DetectionResult(
            probability_map=probability_map,
            mean_ai_probability=final_probability,
            simple_mean_ai_probability=float(pixel_map.mean()),
            high_evidence_fraction=float(np.mean(probability_map >= threshold)),
            regional_evidence=regional_evidence,
            region_threshold=threshold,
            detected_regions=region_count,
            overlay=overlay,
            rows=rows,
            columns=columns,
            analysis_windows=window_count,
            scales=scales,
            probability_is_calibrated=calibrated,
        )


def _axis_positions(length: int, scale: int) -> list[int]:
    if length <= scale:
        return [0]
    stride = max(1, scale // 2)
    positions = list(range(0, length - scale + 1, stride))
    if positions[-1] != length - scale:
        positions.append(length - scale)
    return positions


def _analysis_windows(
    image: np.ndarray, scale: int
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """Cria recortes contextuais sobrepostos e registra sua área na imagem original."""
    height, width = image.shape[:2]
    pad_bottom = max(0, scale - height)
    pad_right = max(0, scale - width)
    border = cv2.BORDER_REFLECT_101 if height > 1 and width > 1 else cv2.BORDER_REPLICATE
    padded = cv2.copyMakeBorder(image, 0, pad_bottom, 0, pad_right, border)
    windows = []
    regions = []
    for top in _axis_positions(height, scale):
        for left in _axis_positions(width, scale):
            crop = padded[top : top + scale, left : left + scale]
            windows.append(cv2.resize(crop, (BLOCK_SIZE, BLOCK_SIZE), interpolation=cv2.INTER_AREA))
            regions.append((top, left, min(top + scale, height), min(left + scale, width)))
    return np.asarray(windows), regions


def create_overlay(
    image: np.ndarray,
    probability_map: np.ndarray,
    alpha: float,
    region_labels: np.ndarray | None = None,
) -> Image.Image:
    height, width = image.shape[:2]
    grid = np.clip(np.rint(probability_map * 255), 0, 255).astype(np.uint8)
    expanded = cv2.resize(grid, (width, height), interpolation=cv2.INTER_NEAREST)
    heatmap = cv2.cvtColor(cv2.applyColorMap(expanded, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted(image, 1.0 - alpha, heatmap, alpha, 0)
    # Linhas discretas tornam visivel qual area originou cada previsao.
    for x in range(BLOCK_SIZE, width, BLOCK_SIZE):
        cv2.line(blended, (x, 0), (x, height), (255, 255, 255), 1)
    for y in range(BLOCK_SIZE, height, BLOCK_SIZE):
        cv2.line(blended, (0, y), (width, y), (255, 255, 255), 1)
    if region_labels is not None:
        for label in range(1, int(region_labels.max()) + 1):
            mask = (region_labels == label).astype(np.uint8) * 255
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(blended, contours, -1, (255, 255, 255), 3)
            cv2.drawContours(blended, contours, -1, (20, 20, 20), 1)
    return Image.fromarray(blended)


def find_model_dir(root: Path) -> Path:
    candidates = (
        root / "models",
        root / "Cod" / "models",
        root.parent / "Processado" / root.name / "Cod" / "models",
    )
    for directory in candidates:
        if (directory / "svm_robusto_fusion.pkl").exists() or (directory / "svm_fusion.pkl").exists():
            return directory
    return candidates[1]
