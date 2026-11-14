from pathlib import Path
from io import BytesIO
import sys

# Modelos v3 (ThunderSVM) exigem o .venv do projeto — não o Python do sistema.
_VENV_OK = Path(sys.prefix).resolve() == (Path(__file__).resolve().parent / ".venv").resolve()
if not _VENV_OK:
    sys.stderr.write(
        "\n[VerifAI] Use o Python do projeto:\n"
        "  ./run_app.sh\n"
        "  ou:  .venv/bin/streamlit run app.py\n"
        f"(agora: {sys.executable})\n\n"
    )
    raise SystemExit(1)

import pandas as pd
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

from app.detector import SyntheticImageDetector, classify_verdict, find_model_dir

ROOT = Path(__file__).resolve().parent
FAMILY_LABELS = {
    "fusion": "Fusão (FFT+HOG+LBP)",
    "fft": "Só FFT",
    "hog": "Só HOG",
}
VERDICT_META = {
    "ai": ("PROVÁVEL IA", "#ff3d57", ""),
    "real": ("PROVÁVEL REAL", "#00e676", " real"),
    "inconclusive": ("INCONCLUSIVO (40–60%)", "#ffb300", " gray"),
}

st.set_page_config(
    page_title="VerifAI · Scanner Forense",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(
    """
<style>
:root{--bg:#0f131b;--lowest:#0a0e15;--low:#181c23;--panel:#1c2027;--high:#262a32;--text:#dfe2ed;--muted:#849396;--cyan:#00e5ff;--cyan2:#c3f5ff;--red:#ff3d57;--green:#00e676;--amber:#ffb300}
html,body,[class*="css"]{font-family:Inter,Geist,system-ui,sans-serif}.stApp{background:var(--bg);color:var(--text)}[data-testid="stHeader"],#MainMenu,footer,[data-testid="stSidebar"]{display:none!important}.block-container{padding:0 2.4rem 1rem!important;max-width:1360px!important}
.topbar{height:58px;margin:0 -2.4rem;padding:0 2.4rem;display:flex;align-items:center;justify-content:space-between;background:#0a0e15f5;border-bottom:1px solid #00e5ff1c;box-shadow:0 3px 18px #0008}.brand-wrap,.brand,.engine,.intro,.card-title{display:flex;align-items:center}.brand-wrap{gap:20px}.brand{gap:9px;font:700 20px 'Space Grotesk',sans-serif;color:var(--cyan2)}.brand-icon{width:31px;height:31px;display:grid;place-items:center;border-radius:7px;background:var(--high);color:var(--cyan);box-shadow:0 0 16px #00e5ff35}.eyebrow{font:600 10px monospace;letter-spacing:.11em;text-transform:uppercase}.engine{gap:7px;color:var(--muted);font:600 10px monospace}.dot{width:7px;height:7px;border-radius:50%;background:var(--green);box-shadow:0 0 9px var(--green)}
.intro{justify-content:space-between;padding:18px 0 14px}
html,body{overflow:hidden}.stApp{height:100vh;overflow:hidden}.block-container{padding:0 1.5rem .5rem!important;max-width:1600px!important}
.topbar{height:55px;margin:0 -1.5rem;padding:0 1.5rem}.engine{color:var(--text);background:var(--low);border:1px solid #ffffff12;border-radius:99px;padding:5px 9px}
.intro{padding:13px 0 11px}.intro>div:first-child{display:flex;align-items:center;gap:10px}.eyebrow{margin:0;background:var(--high);border:1px solid #ffffff12;border-radius:99px;padding:5px 9px;color:var(--cyan)}.intro h1{margin:0;font:600 22px/1.08 'Space Grotesk',sans-serif;color:var(--cyan2);letter-spacing:-.035em;white-space:nowrap}.intro p{margin:0 0 0 auto;color:#bac9cc;font-size:13px}
[data-testid="stFileUploader"] section{min-height:105px!important;border-color:#849396!important;background:transparent!important;border:1px dashed #849396!important;border-radius:6px!important}
.upload-caption{text-align:center;color:var(--muted);font:600 10px monospace;letter-spacing:.08em;margin-bottom:5px}
.image-label{color:var(--cyan);font:600 10px monospace;letter-spacing:.08em;text-transform:uppercase;margin:0 0 5px}
[data-testid="stButton"]{position:fixed!important;left:auto!important;right:1.5rem!important;top:13px!important;width:auto!important;min-width:0!important;z-index:9999}[data-testid="stButton"]>div{width:auto!important}[data-testid="stButton"] button{width:auto!important;min-width:125px!important;height:32px!important;padding:0 14px!important;background:var(--cyan)!important;color:#00363d!important;border:0!important;border-radius:4px!important;font-weight:700!important;box-shadow:0 0 18px #00e5ff3d}
.confidence{background:var(--panel);border-left:4px solid var(--red);border-radius:5px;padding:13px}.confidence.real{border-left-color:var(--green)}.confidence.gray{border-left-color:var(--amber)}
.confidence-value{font:700 38px/1 monospace;color:var(--red)}.confidence.real .confidence-value{color:var(--green)}.confidence.gray .confidence-value{color:var(--amber)}
.confidence-title{font-weight:700;margin-left:8px}.confidence p{color:#bac9cc;font-size:12px;line-height:1.4;margin:8px 0}
.bar{height:7px;background:#31353d;border-radius:99px;overflow:hidden}.bar>div{height:100%;background:var(--red)}.confidence.real .bar>div{background:var(--green)}.confidence.gray .bar>div{background:var(--amber)}
.diagnostic-list{margin-top:12px;background:var(--lowest);border:1px solid #ffffff0c}
.diagnostic-row{display:flex;justify-content:space-between;padding:9px;border-bottom:1px solid #ffffff0c;font-size:12px;color:#bac9cc}.diagnostic-row:last-child{border:0}.diagnostic-row strong{color:var(--cyan);font-family:monospace}
.family-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:10px}
.family-card{background:var(--lowest);border:1px solid #ffffff0c;border-radius:4px;padding:8px;text-align:center}
.family-card.active{border-color:#00e5ff66}.family-card span{display:block;color:var(--muted);font:600 9px monospace;letter-spacing:.08em}
.family-card b{color:var(--cyan);font:700 16px monospace}
div[data-testid="stImage"] img{border-radius:4px;border:1px solid #00e5ff21;max-height:500px;object-fit:contain;background:var(--lowest)}
[data-testid="stExpander"]{background:var(--low)!important;border:1px solid #ffffff08!important;border-radius:8px!important}
@media(max-width:900px){html,body,.stApp{overflow:auto}.engine,.intro p{display:none}.block-container{padding:0 1rem 1rem!important}.topbar{margin:0 -1rem;padding:0 1rem}.intro h1{font-size:20px}.family-grid{grid-template-columns:1fr}}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_detector(path: str):
    return SyntheticImageDetector(Path(path))


def square_preview(source: Image.Image, size: int = 720) -> Image.Image:
    fitted = ImageOps.contain(source.convert("RGB"), (size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), (10, 14, 21))
    canvas.paste(fitted, ((size - fitted.width) // 2, (size - fitted.height) // 2))
    return canvas


def family_cards_html(scores: dict, active: str) -> str:
    cards = []
    for key in ("fusion", "fft", "hog"):
        if key not in scores:
            continue
        label = FAMILY_LABELS[key]
        active_cls = " active" if key == active else ""
        cards.append(
            f'<div class="family-card{active_cls}"><span>{label}</span>'
            f"<b>{scores[key]:.0%}</b></div>"
        )
    return f'<div class="family-grid">{"".join(cards)}</div>'


st.markdown(
    '<div class="topbar"><div class="brand-wrap"><div class="brand">'
    '<span class="brand-icon">◉</span>VerifAI</div>'
    '<div class="engine"><span class="dot"></span> Motor Neural Ativo</div>'
    "</div></div>",
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="intro"><div><div class="eyebrow">◉ Detector Neural de IA</div>'
    "<h1>Verificação de Autenticidade</h1></div>"
    "<p>Imagem inteira (sem CIFAKE / sem grid) · veredito da Fusão.</p></div>",
    unsafe_allow_html=True,
)

try:
    detector = load_detector(str(find_model_dir(ROOT)))
except (FileNotFoundError, ValueError, OSError) as error:
    st.error("Modelo indisponível para a aplicação.")
    st.code(str(error), language=None)
    st.stop()

if "submitted_image" not in st.session_state:
    st.session_state.submitted_image = None

if st.session_state.submitted_image is not None:
    if st.button("↻  Nova Análise", key="new_analysis"):
        st.session_state.submitted_image = None
        st.rerun()
else:
    _, upload_center, _ = st.columns([1, 2.2, 1], gap="medium")
    with upload_center:
        st.markdown(
            '<div class="upload-caption">INGESTÃO DE MÍDIA · ARQUIVO ÚNICO</div>',
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader(
            "Arraste e solte sua foto aqui",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=False,
        )
        with st.expander("⚙ Configurações", expanded=False):
            available = [n for n in ("fusion", "fft", "hog") if n in detector.models]
            family = st.selectbox(
                "Modelo do veredito (padrão = Fusão)",
                available,
                format_func=lambda key: FAMILY_LABELS[key],
                index=0,
            )
            alpha = st.slider("Intensidade do overlay", 0.0, 1.0, 0.45, 0.05)
            st.caption("Foto inteira (letterbox) → SVM. FFT/HOG só para comparar.")
        if uploaded is not None:
            st.session_state.submitted_image = {
                "data": uploaded.getvalue(),
                "name": uploaded.name,
                "family": family,
                "alpha": alpha,
            }
            st.rerun()

if st.session_state.submitted_image is None:
    st.markdown(
        '<div style="text-align:center;color:#849396;font-size:12px;margin-top:18px">'
        "⬡ Processamento local · selecione uma imagem</div>",
        unsafe_allow_html=True,
    )
    st.stop()

submission = st.session_state.submitted_image
family = submission.get("family", "fusion")
alpha = submission["alpha"]
try:
    image = Image.open(BytesIO(submission["data"]))
    image.load()
except (UnidentifiedImageError, OSError) as error:
    st.error(f"Não foi possível abrir a imagem: {error}")
    st.stop()

with st.spinner("Analisando imagem inteira (Fusão)..."):
    result = detector.detect(image, overlay_alpha=alpha, family=family)

score = result.score
status, color, vclass = VERDICT_META[result.verdict]
st.markdown(
    f'<div class="card-title" style="max-width:1250px;margin:8px auto 9px;padding-bottom:8px;'
    f'border-bottom:1px solid #3b494c55"><span style="color:{color}">●</span> '
    "Resultado da Verificação</div>",
    unsafe_allow_html=True,
)

original_column, heatmap_column, diagnostic_column = st.columns([1, 1, 1.08], gap="medium")
with original_column:
    st.markdown('<div class="image-label">Imagem original</div>', unsafe_allow_html=True)
    st.image(square_preview(image), width="stretch")
with heatmap_column:
    st.markdown('<div class="image-label">Leitura (imagem inteira)</div>', unsafe_allow_html=True)
    st.image(square_preview(result.overlay), width="stretch")
with diagnostic_column:
    st.markdown(
        f"""<div class="confidence{vclass}">
            <div><span class="confidence-value">{score:.0%}</span>
            <span class="confidence-title">Chance IA</span></div>
            <p>{status}. {result.decision_strategy}.</p>
            <div class="bar"><div style="width:{score*100:.1f}%"></div></div>
        </div>
        <div class="diagnostic-list">
            <div class="diagnostic-row"><span>Score (modelo ativo)</span><strong>{score:.1%}</strong></div>
            <div class="diagnostic-row"><span>Fusão</span><strong>{result.fusion_score:.1%}</strong></div>
            <div class="diagnostic-row"><span>Entrada</span><strong>imagem inteira (letterbox {result.input_size})</strong></div>
            <div class="diagnostic-row"><span>Modelo ativo</span><strong>{FAMILY_LABELS.get(result.active_family, result.active_family)}</strong></div>
        </div>
        {family_cards_html(result.family_scores, result.active_family)}""",
        unsafe_allow_html=True,
    )

with st.expander("▣ Comparação por família (referência)", expanded=False):
    compare = pd.DataFrame(
        [
            {
                "Família": FAMILY_LABELS.get(key, key),
                "Chance IA": value,
                "Veredito": VERDICT_META[classify_verdict(value)][0],
            }
            for key, value in result.family_scores.items()
        ]
    )
    st.dataframe(
        compare.style.format({"Chance IA": "{:.1%}"}),
        width="stretch",
        hide_index=True,
    )
    cols = st.columns(3)
    cols[0].metric("Dimensões", f"{image.width}×{image.height}")
    cols[1].metric("Modo", "imagem inteira")
    cols[2].metric("SVM (letterbox)", f"{result.input_size}×{result.input_size}")

if not result.probability_is_calibrated:
    st.caption("ℹ Probabilidade estimada pela distância à fronteira do SVM.")
