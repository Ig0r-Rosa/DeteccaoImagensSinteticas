"""VerifAI — inferência: imagem inteira (letterbox) → SVM v3 (fusão)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import warnings

import cv2
import joblib
import numpy as np
from PIL import Image
from sklearn.exceptions import InconsistentVersionWarning

# Mesmo pré-processamento/features do treino (cod/v3).
_COD = Path(__file__).resolve().parents[1] / "cod"
if str(_COD) not in sys.path:
    sys.path.insert(0, str(_COD))
from v3.data import FEATURE_SIZE, letterbox_rgb  # noqa: E402
from v3.features import (  # noqa: E402
    extrair_features_fft,
    extrair_features_hog,
    extrair_features_lbp,
)

MAX_DISPLAY_SIDE = 512
FEATURE_TYPES = ("fft", "hog", "lbp")
FAMILY_ORDER = ("fusion", "fft", "hog")
GRAY_LOW = 0.40
GRAY_HIGH = 0.60


def _load_trusted_artifact(path: Path):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
        try:
            return joblib.load(path)
        except ModuleNotFoundError as erro:
            if "thundersvm" in str(erro):
                raise ModuleNotFoundError(
                    "Modelo v3 foi treinado com ThunderSVM. "
                    "Rode o app pelo .venv:  .venv/bin/streamlit run app.py  "
                    "ou  ./run_app.sh"
                ) from erro
            raise


@dataclass(frozen=True)
class DetectionResult:
    """Resultado de uma passagem na imagem inteira."""

    score: float
    fusion_score: float
    family_scores: dict[str, float]
    active_family: str
    verdict: str
    decision_strategy: str
    overlay: Image.Image
    probability_is_calibrated: bool
    mean_ai_probability: float
    global_ai_probability: float
    input_size: int


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -40, 40)
    return 1.0 / (1.0 + np.exp(-values))


def classify_verdict(score: float) -> str:
    if score < GRAY_LOW:
        return "real"
    if score > GRAY_HIGH:
        return "ai"
    return "inconclusive"


def resize_for_display(rgb: np.ndarray, max_side: int = MAX_DISPLAY_SIDE) -> np.ndarray:
    height, width = rgb.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return rgb
    scale = max_side / float(longest)
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)


def create_overlay(image: np.ndarray, score: float, alpha: float) -> Image.Image:
    """Tinta uniforme — sem grid."""
    level = int(np.clip(round(score * 255), 0, 255))
    heat = np.full(image.shape[:2], level, dtype=np.uint8)
    heatmap = cv2.cvtColor(cv2.applyColorMap(heat, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted(image, 1.0 - alpha, heatmap, alpha, 0)
    return Image.fromarray(blended)


def _detect_model_prefix(model_dir: Path) -> str:
    if (model_dir / "svm_v3_fusion.pkl").exists():
        return "v3_"
    if (model_dir / "svm_robusto_fusion.pkl").exists():
        return "robusto_"
    return ""


def find_model_dir(root: Path) -> Path:
    candidates = (
        root / "models",
        root / "cod" / "models",
        root.parent / "Processado" / root.name / "cod" / "models",
    )
    for directory in candidates:
        if (
            (directory / "svm_v3_fusion.pkl").exists()
            or (directory / "svm_robusto_fusion.pkl").exists()
            or (directory / "svm_fusion.pkl").exists()
        ):
            return directory
    return candidates[1]


class SyntheticImageDetector:
    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        self.prefix = _detect_model_prefix(self.model_dir)
        self.scalers = self._load_scalers()
        self.models = self._load_models()
        self.model = self.models["fusion"]
        self.input_size = FEATURE_SIZE

    def _artifact(self, name: str) -> Path:
        return self.model_dir / name

    def _load_scalers(self) -> dict[str, object]:
        scalers = {}
        for kind in FEATURE_TYPES:
            path = self._artifact(f"scaler_{self.prefix}{kind}.pkl")
            if not path.exists():
                raise FileNotFoundError(f"Scaler ausente: {path}")
            scalers[kind] = _load_trusted_artifact(path)
        return scalers

    def _load_models(self) -> dict[str, object]:
        models = {}
        for family in FAMILY_ORDER:
            path = self._artifact(f"svm_{self.prefix}{family}.pkl")
            if not path.exists() and family == "fusion" and not self.prefix:
                path = self._artifact("svm_fusion.pkl")
            if not path.exists():
                if family == "fusion":
                    raise FileNotFoundError(f"Modelo ausente: {path}")
                continue
            models[family] = self._normalize_model(_load_trusted_artifact(path))
        if "fusion" not in models:
            raise FileNotFoundError("Modelo de fusão obrigatório não encontrado.")
        return models

    @staticmethod
    def _normalize_model(model):
        if not hasattr(model, "classes_"):
            model.classes_ = np.asarray([0, 1])
        return model

    def _scaled_parts(self, blocks: np.ndarray) -> dict[str, np.ndarray]:
        extractors = {
            "fft": extrair_features_fft,
            "hog": extrair_features_hog,
            "lbp": extrair_features_lbp,
        }
        parts = {}
        for kind in FEATURE_TYPES:
            raw = np.asarray([extractors[kind](block) for block in blocks])
            parts[kind] = self.scalers[kind].transform(raw)
        return parts

    def _features_for_family(self, parts: dict[str, np.ndarray], family: str) -> np.ndarray:
        if family == "fusion":
            return np.hstack([parts[kind] for kind in FEATURE_TYPES])
        return parts[family]

    def _ai_probabilities(self, model, features: np.ndarray) -> tuple[np.ndarray, bool]:
        """Chance de IA (FAKE=0), robusta à ordem das colunas do ThunderSVM."""
        pred = np.asarray(model.predict(features), dtype=int).ravel()
        if hasattr(model, "predict_proba") and getattr(model, "probability", True):
            proba = np.asarray(model.predict_proba(features), dtype=float)
            confidence = proba.max(axis=1)
            ai_score = np.where(pred == 0, confidence, 1.0 - confidence)
            return ai_score, True
        decision = np.asarray(model.decision_function(features), dtype=float).ravel()
        strength = _sigmoid(np.abs(decision))
        ai_score = np.where(pred == 0, strength, 1.0 - strength)
        return ai_score, False

    def _score_full_image(self, rgb: np.ndarray, family: str) -> tuple[float, bool]:
        """Imagem inteira (letterbox) → features → SVM."""
        whole = letterbox_rgb(rgb, FEATURE_SIZE)
        parts = self._scaled_parts(np.asarray([whole]))
        features = self._features_for_family(parts, family)
        probs, calibrated = self._ai_probabilities(self.models[family], features)
        return float(probs[0]), calibrated

    def detect(
        self,
        image: Image.Image,
        overlay_alpha: float = 0.45,
        family: str = "fusion",
        scales: tuple[int, ...] | None = None,
    ) -> DetectionResult:
        """Uma passagem na foto toda (sem grid)."""
        del scales
        if family not in self.models:
            raise ValueError(f"Família indisponível: {family}. Opções: {sorted(self.models)}")

        original = np.asarray(image.convert("RGB"))
        display = resize_for_display(original)

        family_scores: dict[str, float] = {}
        calibrated = False
        for name in FAMILY_ORDER:
            if name not in self.models:
                continue
            prob, ok = self._score_full_image(original, name)
            family_scores[name] = prob
            calibrated = calibrated or ok

        score = float(family_scores[family])
        fusion_score = float(family_scores.get("fusion", score))
        strategy = f"imagem inteira · {family}"

        return DetectionResult(
            score=score,
            fusion_score=fusion_score,
            family_scores=family_scores,
            active_family=family,
            verdict=classify_verdict(score),
            decision_strategy=strategy,
            overlay=create_overlay(display, score, overlay_alpha),
            probability_is_calibrated=calibrated,
            mean_ai_probability=score,
            global_ai_probability=score,
            input_size=FEATURE_SIZE,
        )
