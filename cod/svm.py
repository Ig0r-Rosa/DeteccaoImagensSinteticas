"""Treino SVM: ThunderSVM (GPU) → sklearn RBF → aproximação."""

from __future__ import annotations

import gc
from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.kernel_approximation import RBFSampler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC

MAX_RBF_SAMPLES = 20000
MAX_THUNDER_SAMPLES = 40000


def _liberar_gpu() -> None:
    gc.collect()
    try:
        import cupy

        cupy.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass


def _thundersvm_ok() -> bool:
    try:
        import thundersvm  # noqa: F401

        return True
    except Exception:
        return False


def _peso_classes(peso_real: float = 2.0) -> dict[int, float]:
    """Pesa REAL (1) para reduzir falso positivo no app."""
    return {0: 1.0, 1: float(peso_real)}


def _criar_thundersvm(C: float = 10.0, seed: int = 42, peso_real: float = 2.0):
    from thundersvm import SVC as ThunderSVC

    return ThunderSVC(
        kernel="rbf",
        C=C,
        gamma="auto",
        probability=True,
        class_weight=_peso_classes(peso_real),
        random_state=seed,
        gpu_id=0,
        max_mem_size=3500,
    )


def _criar_sklearn_rbf(C: float = 10.0, seed: int = 42, peso_real: float = 2.0):
    return SVC(
        kernel="rbf",
        C=C,
        gamma="scale",
        probability=True,
        random_state=seed,
        cache_size=2000,
        class_weight=_peso_classes(peso_real),
    )


def _criar_rbf_aproximado(
    C: float = 10.0, seed: int = 42, n_components: int = 1024, peso_real: float = 2.0
):
    """RBF aproximado + LinearSVC calibrado (fallback)."""
    base = Pipeline(
        [
            ("rbf", RBFSampler(gamma="scale", n_components=n_components, random_state=seed)),
            (
                "svm",
                LinearSVC(
                    C=C,
                    max_iter=8000,
                    dual="auto",
                    random_state=seed,
                    class_weight=_peso_classes(peso_real),
                ),
            ),
        ]
    )
    return CalibratedClassifierCV(base, method="sigmoid", cv=3)


def _amostrar(X: np.ndarray, y: np.ndarray, max_n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Amostra estratificada se o treino for grande demais."""
    if len(y) <= max_n:
        return X, y
    X_s, _, y_s, _ = train_test_split(X, y, train_size=max_n, stratify=y, random_state=seed)
    return X_s, y_s


def treinar_svm_seguro(
    X_train: np.ndarray,
    y_train: np.ndarray,
    prefer_gpu: bool = True,
    C: float = 10.0,
    seed: int = 42,
    peso_real: float = 2.0,
) -> tuple[Any, str]:
    """Treina com fallback: GPU → RBF sklearn → aproximação."""
    n = len(y_train)
    _liberar_gpu()

    if prefer_gpu and _thundersvm_ok():
        try:
            X_fit, y_fit = _amostrar(X_train, y_train, MAX_THUNDER_SAMPLES, seed)
            if len(y_fit) < n:
                print(f"  Amostra GPU: {len(y_fit)}/{n}")
            print(f"  backend=thundersvm-rbf (n={len(y_fit)}, peso_real={peso_real})")
            modelo = _criar_thundersvm(C=C, seed=seed, peso_real=peso_real)
            modelo.fit(X_fit, y_fit)
            if not hasattr(modelo, "classes_"):
                modelo.classes_ = np.unique(y_fit)
            return modelo, "thundersvm-rbf"
        except Exception as erro:
            print(f"  ThunderSVM falhou ({erro}); usando sklearn RBF.")
            _liberar_gpu()

    X_fit, y_fit = _amostrar(X_train, y_train, MAX_RBF_SAMPLES, seed)
    if len(y_fit) < n:
        print(f"  Amostra RBF: {len(y_fit)}/{n}")
    try:
        print(f"  backend=sklearn-rbf (n={len(y_fit)}, peso_real={peso_real})")
        modelo = _criar_sklearn_rbf(C=C, seed=seed, peso_real=peso_real)
        modelo.fit(X_fit, y_fit)
        return modelo, "sklearn-rbf"
    except Exception as erro:
        print(f"  sklearn-rbf falhou ({erro}); tentando aproximação.")

    n_comp = 1024 if n <= 80000 else 2048
    print(f"  backend=rbf-approx (n={n}, comps={n_comp})")
    modelo = _criar_rbf_aproximado(C=C, seed=seed, n_components=n_comp, peso_real=peso_real)
    modelo.fit(X_train, y_train)
    return modelo, "rbf-approx+linearsvc"
