"""CSS da interface Streamlit."""

APP_CSS = """
<style>
:root{
  --bg:#0f131b;--lowest:#0a0e15;--low:#181c23;--panel:#1c2027;--high:#262a32;
  --text:#dfe2ed;--muted:#849396;--cyan:#00e5ff;--cyan2:#c3f5ff;
  --red:#ff3d57;--green:#00e676;--amber:#ffb300;--bar-h:52px;
}
html,body,[class*="css"]{font-family:Inter,Geist,system-ui,sans-serif}
html,body,.stApp{
  height:100vh!important;max-height:100vh!important;overflow:hidden!important;
  background:var(--bg)!important;color:var(--text)
}
[data-testid="stHeader"],#MainMenu,footer,[data-testid="stSidebar"],
[data-testid="stDecoration"],[data-testid="stToolbar"],.stDeployButton,
[data-testid="stStatusWidget"]{display:none!important}

.block-container{
  padding:calc(var(--bar-h) + 8px) 1.2rem .5rem!important;
  max-width:1100px!important;height:100vh!important;overflow:hidden!important;
}

/* Topo fixo */
.topbar{
  position:fixed;top:0;left:0;right:0;z-index:200;
  height:var(--bar-h);box-sizing:border-box;
  display:flex;align-items:center;justify-content:center;
  background:#0a0e15;border-bottom:1px solid #00e5ff1c;width:100%
}
.brand{
  display:flex;align-items:center;gap:9px;
  font:700 18px 'Space Grotesk',sans-serif;color:var(--cyan2)
}
.brand-icon{
  width:28px;height:28px;display:grid;place-items:center;border-radius:7px;
  background:var(--high);color:var(--cyan);box-shadow:0 0 14px #00e5ff35
}

/* Botão "Nova Análise" na topbar */
div[data-testid="stButton"]{
  position:fixed!important;top:6px!important;right:1.25rem!important;
  width:auto!important;min-width:0!important;height:0!important;
  z-index:210!important;margin:0!important;padding:0!important;overflow:visible!important
}
div[data-testid="stButton"] > div{width:auto!important}
div[data-testid="stButton"] button{
  height:32px!important;min-width:118px!important;padding:0 14px!important;
  background:var(--cyan)!important;color:#00363d!important;border:0!important;
  border-radius:6px!important;font:700 12px/1 Inter,sans-serif!important;
  box-shadow:0 0 16px #00e5ff3d
}

/* Upload: área grande; input cobre tudo para drag/click */
[data-testid="stFileUploader"]{margin:0!important}
[data-testid="stFileUploader"] [data-testid="stWidgetLabel"]{display:none!important}
[data-testid="stFileUploaderDropzoneInstructions"],
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] small{display:none!important}
[data-testid="stFileUploaderDropzone"]{
  position:relative!important;min-height:50vh!important;height:50vh!important;
  border:1px dashed #849396!important;background:transparent!important;
  border-radius:10px!important;
  display:flex!important;align-items:center!important;justify-content:center!important;
  gap:0!important;padding:0!important;overflow:hidden!important
}
[data-testid="stFileUploaderDropzoneInput"]{
  position:absolute!important;inset:0!important;
  width:100%!important;height:100%!important;max-width:none!important;
  opacity:0!important;cursor:pointer!important;z-index:5!important;
  display:block!important;margin:0!important;padding:0!important;border:0!important
}
[data-testid="stFileUploaderDropzone"] button,
[data-testid="stFileUploader"] button{
  display:inline-flex!important;visibility:visible!important;opacity:1!important;
  color:inherit!important;margin:0!important;pointer-events:none!important;z-index:1!important
}
[data-testid="stFileUploaderDropzone"] button span,
[data-testid="stFileUploaderDropzone"] button p,
[data-testid="stFileUploader"] button p{
  display:inline!important;visibility:visible!important;opacity:1!important;
  color:inherit!important;font-size:inherit!important
}

/* Resultado */
div[data-testid="stHorizontalBlock"]:has(img){
  align-items:flex-start!important;gap:0.6rem!important;margin-top:0!important
}
div[data-testid="stImage"]{
  margin:0!important;display:block!important;
  width:100%!important;max-width:450px!important;aspect-ratio:1/1!important
}
div[data-testid="stImage"] img{
  border-radius:10px;border:1px solid #00e5ff40;
  width:100%!important;height:100%!important;
  max-width:450px!important;max-height:450px!important;
  aspect-ratio:1/1!important;object-fit:contain!important;background:var(--lowest);
  box-shadow:0 8px 28px #0006,0 0 24px #00e5ff14
}

.verdict-panel{display:flex;flex-direction:column;gap:12px;max-width:100%;margin:0}
.confidence{
  background:linear-gradient(160deg,var(--panel),var(--lowest));
  border:1px solid #ffffff10;border-left:5px solid var(--red);
  border-radius:10px;padding:18px 16px 14px
}
.confidence.real{border-left-color:var(--green)}
.confidence.gray{border-left-color:var(--amber)}
.confidence-value{font:800 48px/1 monospace;color:var(--red);letter-spacing:-.03em}
.confidence.real .confidence-value{color:var(--green)}
.confidence.gray .confidence-value{color:var(--amber)}
.confidence-title{
  display:inline-block;margin-left:8px;font:700 12px monospace;
  letter-spacing:.06em;text-transform:uppercase;color:#bac9cc;vertical-align:middle
}
.verdict-badge{
  display:inline-block;margin-top:10px;padding:5px 10px;border-radius:4px;
  font:700 12px monospace;letter-spacing:.04em
}
.verdict-badge.ai{background:#ff3d5722;color:var(--red);border:1px solid #ff3d5755}
.verdict-badge.real{background:#00e67622;color:var(--green);border:1px solid #00e67655}
.verdict-badge.gray{background:#ffb30022;color:var(--amber);border:1px solid #ffb30055}
.confidence p{color:#bac9cc;font-size:12px;line-height:1.35;margin:8px 0 10px}
.bar{height:8px;background:#31353d;border-radius:99px;overflow:hidden}
.bar>div{height:100%;background:var(--red);border-radius:99px}
.confidence.real .bar>div{background:var(--green)}
.confidence.gray .bar>div{background:var(--amber)}

.diagnostic-list{background:var(--lowest);border:1px solid #ffffff0c;border-radius:8px}
.diagnostic-row{
  display:flex;justify-content:space-between;align-items:center;
  padding:9px 12px;border-bottom:1px solid #ffffff0c;font-size:12px;color:#bac9cc
}
.diagnostic-row:last-child{border:0}
.diagnostic-row strong{color:var(--cyan);font-family:monospace;font-size:13px}

.family-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}
.family-card{
  background:var(--lowest);border:1px solid #ffffff0c;border-radius:8px;
  padding:10px 6px;text-align:center
}
.family-card.active{border-color:#00e5ff66;box-shadow:0 0 12px #00e5ff22}
.family-card span{
  display:block;color:var(--muted);font:600 9px monospace;
  letter-spacing:.04em;margin-bottom:4px
}
.family-card b{color:var(--cyan);font:700 17px monospace}
.meta-note{color:var(--muted);font-size:11px}

.boot-loader{
  position:fixed;inset:0;z-index:300;display:grid;place-items:center;background:var(--bg)
}
.boot-spin{
  width:34px;height:34px;border-radius:50%;
  border:3px solid #00e5ff28;border-top-color:var(--cyan);
  animation:boot-spin .65s linear infinite;box-shadow:0 0 18px #00e5ff40
}
@keyframes boot-spin{to{transform:rotate(360deg)}}

[data-testid="stExpander"]{
  background:var(--low)!important;border:1px solid #ffffff08!important;border-radius:8px!important
}

@media(max-width:900px){
  html,body,.stApp,.block-container{
    overflow:auto!important;height:auto!important;max-height:none!important
  }
  .block-container{padding:calc(var(--bar-h) + 12px) 1rem 1rem!important}
  div[data-testid="stImage"],div[data-testid="stImage"] img{
    max-width:min(450px,92vw)!important;max-height:min(450px,92vw)!important
  }
  .confidence-value{font-size:40px}
  .family-grid{grid-template-columns:1fr}
}
</style>
"""
