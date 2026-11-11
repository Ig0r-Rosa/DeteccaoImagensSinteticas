from pathlib import Path
from io import BytesIO
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError
from app.detector import SyntheticImageDetector, find_model_dir

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="VerifAI · Scanner Forense", page_icon="◉", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
:root{--bg:#0f131b;--lowest:#0a0e15;--low:#181c23;--panel:#1c2027;--high:#262a32;--text:#dfe2ed;--muted:#849396;--cyan:#00e5ff;--cyan2:#c3f5ff;--red:#ff3d57;--green:#00e676}
html,body,[class*="css"]{font-family:Inter,Geist,system-ui,sans-serif}.stApp{background:var(--bg);color:var(--text)}[data-testid="stHeader"],#MainMenu,footer,[data-testid="stSidebar"]{display:none!important}.block-container{padding:0 2.4rem 1rem!important;max-width:1360px!important}
.topbar{height:58px;margin:0 -2.4rem;padding:0 2.4rem;display:flex;align-items:center;justify-content:space-between;background:#0a0e15f5;border-bottom:1px solid #00e5ff1c;box-shadow:0 3px 18px #0008}.brand-wrap,.brand,.engine,.top-actions,.intro,.telemetry,.card-head,.card-title,.verdict{display:flex;align-items:center}.brand-wrap{gap:20px}.brand{gap:9px;font:700 20px 'Space Grotesk',sans-serif;color:var(--cyan2)}.brand-icon{width:31px;height:31px;display:grid;place-items:center;border-radius:7px;background:var(--high);color:var(--cyan);box-shadow:0 0 16px #00e5ff35}.version,.tag,.eyebrow,.technical{font:600 10px monospace;letter-spacing:.11em;text-transform:uppercase}.version{color:var(--cyan);background:var(--high);padding:3px 6px}.engine{gap:7px;color:var(--muted);font:600 10px monospace}.dot{width:7px;height:7px;border-radius:50%;background:var(--green);box-shadow:0 0 9px var(--green)}.top-actions{gap:9px}.top-chip{padding:8px 12px;border-radius:4px;background:var(--high);color:#bac9cc;font-size:12px}.top-chip.primary{background:var(--cyan);color:#00363d;font-weight:700;box-shadow:0 0 18px #00e5ff42}
.intro{justify-content:space-between;padding:18px 0 14px}.eyebrow{color:var(--cyan);margin-bottom:5px}.intro h1{margin:0;font:600 30px/1.08 'Space Grotesk',sans-serif;color:var(--cyan2);letter-spacing:-.035em}.intro p{margin:7px 0 0;color:#bac9cc;font-size:13px}.telemetry{background:var(--low);border:1px solid #ffffff08;border-radius:7px;overflow:hidden}.telemetry>div{padding:8px 14px;border-left:1px solid #ffffff0b}.telemetry>div:first-child{border-left:0}.telemetry span{display:block;color:var(--muted);font:600 9px monospace;letter-spacing:.1em}.telemetry b{color:var(--cyan);font:600 13px monospace}
.section-card{background:var(--low);border:1px solid #ffffff08;border-radius:9px;padding:13px;box-shadow:0 10px 26px #0005;margin-bottom:12px}.card-head{justify-content:space-between;margin-bottom:10px}.card-title{gap:8px;font:600 14px 'Space Grotesk',sans-serif}.card-icon{color:var(--cyan);font-size:17px}.tag{color:var(--muted);background:var(--panel);padding:3px 6px;border-radius:2px}
[data-testid="stFileUploader"]{background:var(--lowest);border-radius:6px;padding:8px}[data-testid="stFileUploader"] section{background:transparent!important;border:1px dashed #00e5ff5c!important;border-radius:6px!important;min-height:150px!important}[data-testid="stFileUploader"] button{background:var(--cyan)!important;color:#00363d!important;border:0!important;border-radius:3px!important;font-weight:700!important;box-shadow:0 0 15px #00e5ff32}[data-testid="stFileUploader"] small,[data-testid="stFileUploaderDropzoneInstructions"] span{color:#bac9cc!important}.privacy{display:flex;gap:9px;padding:9px;background:var(--panel);border-radius:4px;color:#bac9cc;font-size:11px;line-height:1.4}.privacy b{color:var(--green)}
[data-testid="stExpander"]{background:var(--low)!important;border:1px solid #ffffff08!important;border-radius:8px!important}[data-testid="stExpander"] summary{color:var(--text)!important;font-weight:600!important}.viewport{background:var(--lowest);border-radius:6px;border:1px solid #ffffff09;padding:8px}.viewport-meta{display:flex;justify-content:space-between;color:var(--muted);font:600 9px monospace;letter-spacing:.08em;margin-bottom:7px}.viewport-meta strong{color:var(--cyan)}div[data-testid="stImage"] img{border-radius:4px;border:1px solid #00e5ff21;max-height:500px;object-fit:contain;background:var(--lowest)}[data-testid="stRadio"]>div{gap:4px!important}[data-testid="stRadio"] label{background:var(--panel);padding:5px 10px!important;border-radius:3px}[data-testid="stRadio"] label:has(input:checked){background:var(--high);color:var(--cyan)}
.score-card{display:grid;grid-template-columns:125px 1fr;gap:18px;align-items:center;background:var(--low);border:1px solid #ffffff08;border-radius:9px;padding:14px;box-shadow:0 10px 26px #0005}.gauge{--score:0;--gauge-color:var(--red);width:116px;height:116px;border-radius:50%;display:grid;place-items:center;background:conic-gradient(var(--gauge-color) calc(var(--score)*1%),var(--high) 0);box-shadow:0 0 22px #0008;position:relative}.gauge:before{content:"";position:absolute;inset:10px;background:var(--low);border-radius:50%}.gauge-content{position:relative;text-align:center}.gauge-value{font:700 27px monospace}.gauge-label{color:var(--muted);font:600 8px monospace;letter-spacing:.1em}.verdict{gap:8px;background:#93000a;color:#ffdad6;padding:7px 9px;border-radius:3px;font:600 11px monospace;letter-spacing:.06em}.verdict.real{background:#00552b;color:#b9f6ca}.score-card p{color:#bac9cc;font-size:12px;margin:8px 0}.technical{color:var(--muted)}.technical strong{color:var(--cyan);margin-left:5px}
.empty-state{height:470px;background:var(--lowest);border-radius:6px;display:grid;place-items:center;text-align:center;color:var(--muted)}.empty-state .radar{color:var(--cyan);font-size:46px;text-shadow:0 0 25px #00e5ff88}.empty-state b{display:block;color:var(--text);margin:8px 0 3px}[data-testid="stMetric"]{background:var(--panel);padding:8px;border-radius:4px}[data-testid="stMetricValue"]{color:var(--cyan)}
/* Layout compacto do segundo design: uma única viewport, sem painel lateral. */
html,body{overflow:hidden}.stApp{height:100vh;overflow:hidden}.block-container{padding:0 1.5rem .5rem!important;max-width:1600px!important}
.topbar{height:55px;margin:0 -1.5rem;padding:0 1.5rem}.version{display:none}.engine{color:var(--text);background:var(--low);border:1px solid #ffffff12;border-radius:99px;padding:5px 9px}.top-chip:not(.primary){display:none}
.intro{padding:13px 0 11px}.intro>div:first-child{display:flex;align-items:center;gap:10px}.eyebrow{margin:0;background:var(--high);border:1px solid #ffffff12;border-radius:99px;padding:5px 9px}.intro h1{font-size:22px;white-space:nowrap}.intro p{margin:0 0 0 auto}.telemetry{display:none}
[data-testid="stFileUploader"] section{min-height:105px!important;border-color:#849396!important}
.upload-caption{text-align:center;color:var(--muted);font:600 10px monospace;letter-spacing:.08em;margin-bottom:5px}.result-shell{max-width:1250px;margin:0 auto}.image-label{color:var(--cyan);font:600 10px monospace;letter-spacing:.08em;text-transform:uppercase;margin:0 0 5px}
[data-testid="stButton"]{position:fixed!important;left:auto!important;right:1.5rem!important;top:13px!important;width:auto!important;min-width:0!important;z-index:9999}[data-testid="stButton"]>div{width:auto!important}[data-testid="stButton"] button{width:auto!important;min-width:125px!important;height:32px!important;padding:0 14px!important;background:var(--cyan)!important;color:#00363d!important;border:0!important;border-radius:4px!important;font-weight:700!important;box-shadow:0 0 18px #00e5ff3d}
.confidence{background:var(--panel);border-left:4px solid var(--red);border-radius:5px;padding:13px}.confidence.real{border-left-color:var(--green)}.confidence-value{font:700 38px/1 monospace;color:var(--red)}.confidence.real .confidence-value{color:var(--green)}.confidence-title{font-weight:700;margin-left:8px}.confidence p{color:#bac9cc;font-size:12px;line-height:1.4;margin:8px 0}.bar{height:7px;background:#31353d;border-radius:99px;overflow:hidden}.bar>div{height:100%;background:var(--red)}.confidence.real .bar>div{background:var(--green)}.diagnostic-list{margin-top:12px;background:var(--lowest);border:1px solid #ffffff0c}.diagnostic-row{display:flex;justify-content:space-between;padding:9px;border-bottom:1px solid #ffffff0c;font-size:12px;color:#bac9cc}.diagnostic-row:last-child{border:0}.diagnostic-row strong{color:var(--cyan);font-family:monospace}
@media(max-width:900px){html,body,.stApp{overflow:auto}.telemetry,.engine,.intro p,.top-actions .top-chip:not(.primary){display:none}.block-container{padding:0 1rem 1rem!important}.topbar{margin:0 -1rem;padding:0 1rem}.intro h1{font-size:20px}.intro>div:first-child{flex-wrap:wrap}}
</style>""", unsafe_allow_html=True)

@st.cache_resource
def load_detector(path: str): return SyntheticImageDetector(Path(path))

def square_preview(source: Image.Image, size: int = 720) -> Image.Image:
    fitted = ImageOps.contain(source.convert("RGB"), (size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), (10, 14, 21))
    canvas.paste(fitted, ((size-fitted.width)//2, (size-fitted.height)//2))
    return canvas

st.markdown('''<div class="topbar"><div class="brand-wrap"><div class="brand"><span class="brand-icon">◉</span>VerifAI</div><div class="engine"><span class="dot"></span> Motor Neural Ativo</div></div></div>''', unsafe_allow_html=True)
st.markdown('''<div class="intro"><div><div class="eyebrow">◉ Detector Neural de IA</div><h1>Verificação de Autenticidade</h1></div><p>Detecção por frequência, gradientes e microtextura.</p></div>''', unsafe_allow_html=True)

try: detector = load_detector(str(find_model_dir(ROOT)))
except (FileNotFoundError, ValueError, OSError) as error:
    st.error("Modelo indisponível para a aplicação."); st.code(str(error), language=None); st.stop()

if "submitted_image" not in st.session_state:
    st.session_state.submitted_image = None

if st.session_state.submitted_image is not None:
    if st.button("↻  Nova Análise", key="new_analysis"):
        st.session_state.submitted_image = None
        st.rerun()
else:
    upload_left, upload_center, upload_right = st.columns([1, 2.2, 1], gap="medium")
    with upload_center:
        st.markdown('<div class="upload-caption">INGESTÃO DE MÍDIA · ARQUIVO ÚNICO</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Arraste e solte sua foto aqui", type=["jpg","jpeg","png","webp"], accept_multiple_files=False)
        with st.expander("⚙ Configurações de Detecção", expanded=False):
            selected_scales = st.multiselect("Escalas de análise", [32,64,96,128,192,256], default=[32,64,128])
            alpha = st.slider("Intensidade do mapa de calor", 0.0, 1.0, 0.5, 0.05)
        if uploaded is not None:
            st.session_state.submitted_image = {
                "data": uploaded.getvalue(),
                "name": uploaded.name,
                "scales": tuple(selected_scales),
                "alpha": alpha,
            }
            st.rerun()

if st.session_state.submitted_image is None:
    st.markdown('<div style="text-align:center;color:#849396;font-size:12px;margin-top:18px">⬡ Processamento local e privado · selecione uma imagem para iniciar</div>', unsafe_allow_html=True)
    st.stop()

submission = st.session_state.submitted_image
selected_scales = submission["scales"]
alpha = submission["alpha"]
if not selected_scales:
    st.warning("Selecione pelo menos uma escala de análise em uma nova análise.")
    st.stop()
try: image=Image.open(BytesIO(submission["data"])); image.load()
except (UnidentifiedImageError,OSError) as error:
    with upload_center: st.error(f"Não foi possível abrir a imagem: {error}")
    st.stop()
with st.spinner("Executando varredura multiescala..."):
    result=detector.detect(image,overlay_alpha=alpha,scales=tuple(selected_scales))

score=result.mean_ai_probability; is_ai=score>=.5
status="SÍNTESE PROVÁVEL" if is_ai else "SEM SÍNTESE DOMINANTE"
color="#ff3d57" if is_ai else "#00e676"
st.markdown(f'<div class="card-title" style="max-width:1250px;margin:8px auto 9px;padding-bottom:8px;border-bottom:1px solid #3b494c55"><span style="color:{color}">●</span> Resultado da Verificação</div>', unsafe_allow_html=True)
vclass="" if is_ai else " real"
original_column, heatmap_column, diagnostic_column = st.columns([1, 1, 1.08], gap="medium")
with original_column:
    st.markdown('<div class="image-label">Imagem original</div>', unsafe_allow_html=True)
    st.image(square_preview(image), width="stretch")
with heatmap_column:
    st.markdown('<div class="image-label">Mapa de evidências</div>', unsafe_allow_html=True)
    st.image(square_preview(result.overlay), width="stretch")
with diagnostic_column:
    st.markdown(f'''<div class="confidence{vclass}"><div><span class="confidence-value">{score:.0%}</span><span class="confidence-title">Chance de ser IA</span></div><p>{status}. O resultado combina evidência global com regiões conectadas do mapa multiescala.</p><div class="bar"><div style="width:{score*100:.1f}%"></div></div></div><div class="diagnostic-list"><div class="diagnostic-row"><span>Média base</span><strong>{result.simple_mean_ai_probability:.1%}</strong></div><div class="diagnostic-row"><span>Regiões detectadas</span><strong>{result.detected_regions}</strong></div><div class="diagnostic-row"><span>Janelas analisadas</span><strong>{result.analysis_windows}</strong></div><div class="diagnostic-row"><span>Limiar adaptativo</span><strong>{result.region_threshold:.1%}</strong></div><div class="diagnostic-row"><span>Grid do mapa</span><strong>{result.rows}×{result.columns}</strong></div></div>''', unsafe_allow_html=True)

with st.expander("▣ Parâmetros forenses inspecionados e valores por bloco",expanded=False):
    cols=st.columns(4); cols[0].metric("Dimensões",f"{image.width}×{image.height}"); cols[1].metric("Janelas",result.analysis_windows); cols[2].metric("Limiar",f"{result.region_threshold:.1%}"); cols[3].metric("Área suspeita",f"{result.high_evidence_fraction:.1%}")
    frame=pd.DataFrame(result.probability_map,index=[f"L{i+1}" for i in range(result.rows)],columns=[f"C{i+1}" for i in range(result.columns)])
    st.dataframe(frame.style.format("{:.1%}").background_gradient(cmap="turbo",vmin=0,vmax=1),width="stretch")
if not result.probability_is_calibrated: st.caption("ℹ Probabilidade estimada pela distância à fronteira do SVM; calibração pendente para o próximo reteste.")
