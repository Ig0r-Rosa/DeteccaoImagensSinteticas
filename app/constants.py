"""Constantes da interface VerifAI."""

FAMILY_LABELS = {
    "fusion": "Fusão (FFT+HOG+LBP)",
    "fft": "Só FFT",
    "hog": "Só HOG",
    "lbp": "Só LBP",
}

# Ordem exibida no seletor e nos cards.
FAMILY_KEYS = ("fusion", "fft", "hog", "lbp")

# (texto do veredito, classe CSS)
VERDICT_META = {
    "ai": ("PROVÁVEL IA", ""),
    "real": ("PROVÁVEL REAL", " real"),
    "inconclusive": ("INCONCLUSIVO (40–60%)", " gray"),
}

PREVIEW_SIZE = 450
BOOT_SPINNER = (
    '<div class="boot-loader"><div class="boot-spin" aria-label="Carregando"></div></div>'
)
