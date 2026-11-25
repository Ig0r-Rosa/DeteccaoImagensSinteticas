"""Entrada do VerifAI (Streamlit)."""

from pathlib import Path
import sys

# Exige o .venv do projeto (ThunderSVM).
_ROOT = Path(__file__).resolve().parent
if Path(sys.prefix).resolve() != (_ROOT / ".venv").resolve():
    sys.stderr.write(
        "\n[VerifAI] Use o Python do projeto:\n"
        "  ./run_app.sh\n"
        "  ou:  .venv/bin/streamlit run app.py\n"
        f"(agora: {sys.executable})\n\n"
    )
    raise SystemExit(1)

import streamlit as st

from app.detector import SyntheticImageDetector, find_model_dir
from app.ui import apply_theme, render_result, render_topbar, render_upload, show_spinner


@st.cache_resource
def load_detector(path: str) -> SyntheticImageDetector:
    """Carrega os SVMs uma vez e reutiliza."""
    return SyntheticImageDetector(Path(path))


def main() -> None:
    apply_theme()

    # Carrega modelos com tela só da bolinha.
    boot = st.empty()
    show_spinner(boot)
    try:
        detector = load_detector(str(find_model_dir(_ROOT)))
    except (FileNotFoundError, ValueError, OSError) as error:
        boot.empty()
        st.error("Modelo indisponível para a aplicação.")
        st.code(str(error), language=None)
        st.stop()
    boot.empty()

    render_topbar()

    if "submitted_image" not in st.session_state:
        st.session_state.submitted_image = None

    if st.session_state.submitted_image is None:
        render_upload(detector)
        st.stop()

    render_result(detector)


main()
