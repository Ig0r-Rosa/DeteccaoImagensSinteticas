"""Inferência: imagem inteira (letterbox) → SVM v3."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import warnings

import joblib
import numpy as np
from PIL import Image
from sklearn.exceptions import InconsistentVersionWarning

# Features iguais às do treino (cod/).
_COD = Path(__file__).resolve().parents[1] / "cod"
if str(_COD) not in sys.path:
    sys.path.insert(0, str(_COD))
from data import FEATURE_SIZE, letterbox_rgb  # noqa: E402
from features import (  # noqa: E402
    extrair_features_fft,
    extrair_features_hog,
    extrair_features_lbp,
)

FEATURE_TYPES = ("fft", "hog", "lbp")
FAMILY_ORDER = ("fusion", "fft", "hog", "lbp")
GRAY_LOW = 0.40
GRAY_HIGH = 0.60

_EXTRACTORS = {
    "fft": extrair_features_fft,
    "hog": extrair_features_hog,
    "lbp": extrair_features_lbp,
}


def _load_artifact(path: Path):
    """Carrega .pkl e avisa se faltar ThunderSVM."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
        try:
            return joblib.load(path)
        except ModuleNotFoundError as erro:
            if "thundersvm" in str(erro):
                raise ModuleNotFoundError(
                    "Modelo v3 precisa do ThunderSVM. Use ./run_app.sh"
                ) from erro
            raise


def classify_verdict(score: float) -> str:
    """Converte score em real / IA / inconclusivo."""
    if score < GRAY_LOW:
        return "real"
    if score > GRAY_HIGH:
        return "ai"
    return "inconclusive"


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -40, 40)
    return 1.0 / (1.0 + np.exp(-values))


def _model_prefix(model_dir: Path) -> str:
    """Descobre o prefixo dos arquivos (v3_, robusto_ ou vazio)."""
    if (model_dir / "svm_v3_fusion.pkl").exists():
        return "v3_"
    if (model_dir / "svm_robusto_fusion.pkl").exists():
        return "robusto_"
    return ""


def find_model_dir(root: Path) -> Path:
    """Procura a pasta com os .pkl do SVM."""
    candidates = (
        root / "models",
        root / "cod" / "models",
        root.parent / "Processado" / root.name / "cod" / "models",
    )
    for directory in candidates:
        if any(
            (directory / name).exists()
            for name in ("svm_v3_fusion.pkl", "svm_robusto_fusion.pkl", "svm_fusion.pkl")
        ):
            return directory
    return candidates[1]


@dataclass(frozen=True)
class DetectionResult:
    """Saída de uma análise."""

    score: float
    fusion_score: float
    family_scores: dict[str, float]
    active_family: str
    verdict: str
    probability_is_calibrated: bool
    input_size: int


class SyntheticImageDetector:
    """Carrega SVMs e classifica uma imagem inteira."""

    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        self.prefix = _model_prefix(self.model_dir)
        self.scalers = self._load_scalers()
        self.models = self._load_models()
        self.input_size = FEATURE_SIZE

    def _path(self, name: str) -> Path:
        return self.model_dir / name

    def _load_scalers(self) -> dict[str, object]:
        """Carrega um scaler por tipo de feature."""
        scalers = {}
        for kind in FEATURE_TYPES:
            path = self._path(f"scaler_{self.prefix}{kind}.pkl")
            if not path.exists():
                raise FileNotFoundError(f"Scaler ausente: {path}")
            scalers[kind] = _load_artifact(path)
        return scalers

    def _load_models(self) -> dict[str, object]:
        """Carrega fusion (obrigatório) e famílias avulsas se existirem."""
        models = {}
        for family in FAMILY_ORDER:
            path = self._path(f"svm_{self.prefix}{family}.pkl")
            if not path.exists() and family == "fusion" and not self.prefix:
                path = self._path("svm_fusion.pkl")
            if not path.exists():
                if family == "fusion":
                    raise FileNotFoundError(f"Modelo ausente: {path}")
                continue
            models[family] = self._normalize_model(_load_artifact(path))
        return models

    @staticmethod
    def _normalize_model(model):
        """Garante classes_ para modelos ThunderSVM."""
        if not hasattr(model, "classes_"):
            model.classes_ = np.asarray([0, 1])
        return model

    def _scaled_parts(self, blocks: np.ndarray) -> dict[str, np.ndarray]:
        """Extrai e normaliza FFT/HOG/LBP de cada bloco."""
        parts = {}
        for kind in FEATURE_TYPES:
            raw = np.asarray([_EXTRACTORS[kind](block) for block in blocks])
            parts[kind] = self.scalers[kind].transform(raw)
        return parts

    def _features_for_family(self, parts: dict[str, np.ndarray], family: str) -> np.ndarray:
        """Monta o vetor usado pelo SVM da família."""
        if family == "fusion":
            return np.hstack([parts[kind] for kind in FEATURE_TYPES])
        return parts[family]

    def _ai_probabilities(self, model, features: np.ndarray) -> tuple[np.ndarray, bool]:
        """Chance de IA (classe FAKE=0)."""
        pred = np.asarray(model.predict(features), dtype=int).ravel()
        if hasattr(model, "predict_proba") and getattr(model, "probability", True):
            proba = np.asarray(model.predict_proba(features), dtype=float)
            confidence = proba.max(axis=1)
            return np.where(pred == 0, confidence, 1.0 - confidence), True
        decision = np.asarray(model.decision_function(features), dtype=float).ravel()
        strength = _sigmoid(np.abs(decision))
        return np.where(pred == 0, strength, 1.0 - strength), False

    def _score_full_image(self, rgb: np.ndarray, family: str) -> tuple[float, bool]:
        """Letterbox → features → score de um SVM."""
        whole = letterbox_rgb(rgb, FEATURE_SIZE)
        parts = self._scaled_parts(np.asarray([whole]))
        features = self._features_for_family(parts, family)
        probs, calibrated = self._ai_probabilities(self.models[family], features)
        return float(probs[0]), calibrated

    def detect(self, image: Image.Image, family: str = "fusion") -> DetectionResult:
        """Analisa a foto com todos os modelos e devolve o veredito da família ativa."""
        if family not in self.models:
            raise ValueError(f"Família indisponível: {family}. Opções: {sorted(self.models)}")

        original = np.asarray(image.convert("RGB"))
        family_scores: dict[str, float] = {}
        calibrated = False
        for name in FAMILY_ORDER:
            if name not in self.models:
                continue
            prob, ok = self._score_full_image(original, name)
            family_scores[name] = prob
            calibrated = calibrated or ok

        score = float(family_scores[family])
        return DetectionResult(
            score=score,
            fusion_score=float(family_scores.get("fusion", score)),
            family_scores=family_scores,
            active_family=family,
            verdict=classify_verdict(score),
            probability_is_calibrated=calibrated,
            input_size=FEATURE_SIZE,
        )
