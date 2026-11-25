"""Telas e HTML do VerifAI."""

from io import BytesIO

import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

from app.constants import (
    BOOT_SPINNER,
    FAMILY_KEYS,
    FAMILY_LABELS,
    PREVIEW_SIZE,
    VERDICT_META,
)
from app.detector import DetectionResult, SyntheticImageDetector
from app.styles import APP_CSS


def apply_theme() -> None:
    """Aplica CSS e configura a página."""
    st.set_page_config(
        page_title="VerifAI",
        page_icon="◉",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(APP_CSS, unsafe_allow_html=True)


def render_topbar() -> None:
    """Barra superior com a marca."""
    st.html(
        '<div class="topbar"><div class="brand">'
        '<span class="brand-icon">◉</span>VerifAI</div></div>'
    )


def show_spinner(slot) -> None:
    """Mostra só a bolinha de carregamento."""
    slot.html(BOOT_SPINNER)


def square_preview(source: Image.Image, size: int = PREVIEW_SIZE) -> Image.Image:
    """Encaixa a foto num quadro quadrado fixo."""
    fitted = ImageOps.contain(source.convert("RGB"), (size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), (10, 14, 21))
    canvas.paste(fitted, ((size - fitted.width) // 2, (size - fitted.height) // 2))
    return canvas


def _badge_class(verdict: str) -> str:
    return {"ai": "ai", "real": "real"}.get(verdict, "gray")


def _family_cards(scores: dict, active: str) -> str:
    """Cards com o score de cada modelo."""
    cards = []
    for key in FAMILY_KEYS:
        if key not in scores:
            continue
        active_cls = " active" if key == active else ""
        label = FAMILY_LABELS[key]
        cards.append(
            f'<div class="family-card{active_cls}"><span>{label}</span>'
            f"<b>{scores[key]:.0%}</b></div>"
        )
    return f'<div class="family-grid">{"".join(cards)}</div>'


def _row(label: str, value: str) -> str:
    return f'<div class="diagnostic-row"><span>{label}</span><strong>{value}</strong></div>'


def _confidence_block(score: float, verdict: str, model: str) -> str:
    """Bloco principal com % e veredito."""
    status, vclass = VERDICT_META[verdict]
    badge = _badge_class(verdict)
    return (
        f'<div class="confidence{vclass}">'
        f'<div><span class="confidence-value">{score:.0%}</span>'
        f'<span class="confidence-title">Chance IA</span></div>'
        f'<span class="verdict-badge {badge}">{status}</span>'
        f"<p>{model}</p>"
        f'<div class="bar"><div style="width:{score * 100:.1f}%"></div></div>'
        f"</div>"
    )


def _diagnostic_list(result: DetectionResult, image: Image.Image, model: str) -> str:
    """Lista de detalhes técnicos."""
    rows = [
        _row("Score (modelo ativo)", f"{result.score:.1%}"),
        _row("Fusão", f"{result.fusion_score:.1%}"),
        _row("Entrada", f"letterbox {result.input_size}"),
        _row("Modelo ativo", model),
        _row("Dimensões", f"{image.width}×{image.height}"),
    ]
    return f'<div class="diagnostic-list">{"".join(rows)}</div>'


def verdict_panel_html(result: DetectionResult, image: Image.Image) -> str:
    """Monta o painel completo de resultado."""
    model = FAMILY_LABELS.get(result.active_family, result.active_family)
    note = ""
    if not result.probability_is_calibrated:
        note = (
            '<div class="meta-note">ℹ Probabilidade estimada pela '
            "distância à fronteira do SVM.</div>"
        )
    return (
        f'<div class="verdict-panel">'
        f"{_confidence_block(result.score, result.verdict, model)}"
        f"{_diagnostic_list(result, image, model)}"
        f"{_family_cards(result.family_scores, result.active_family)}"
        f"{note}</div>"
    )


def render_upload(detector: SyntheticImageDetector) -> None:
    """Tela inicial: upload + escolha do modelo."""
    _, center, _ = st.columns([0.7, 2.86, 0.7], gap="medium")
    with center:
        uploaded = st.file_uploader(
            "upload",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=False,
            label_visibility="collapsed",
        )
        with st.expander("⚙ Configurações", expanded=False):
            available = [key for key in FAMILY_KEYS if key in detector.models]
            family = st.selectbox(
                "Modelo do veredito",
                available,
                format_func=lambda key: FAMILY_LABELS[key],
                index=0,
            )
        if uploaded is not None:
            st.session_state.submitted_image = {
                "data": uploaded.getvalue(),
                "name": uploaded.name,
                "family": family,
            }
            st.rerun()


def render_result(detector: SyntheticImageDetector) -> None:
    """Tela de resultado: imagem + painel de veredito."""
    if st.button("↻  Nova Análise", key="new_analysis"):
        st.session_state.submitted_image = None
        st.rerun()

    submission = st.session_state.submitted_image
    family = submission.get("family", "fusion")
    try:
        image = Image.open(BytesIO(submission["data"]))
        image.load()
    except (UnidentifiedImageError, OSError) as error:
        st.error(f"Não foi possível abrir a imagem: {error}")
        st.stop()

    slot = st.empty()
    show_spinner(slot)
    result = detector.detect(image, family=family)
    slot.empty()

    left, right = st.columns([1, 1.1], gap="medium")
    with left:
        st.image(square_preview(image), width=PREVIEW_SIZE)
    with right:
        st.html(verdict_panel_html(result, image))
