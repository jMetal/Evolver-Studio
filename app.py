"""Evolver-Studio entrypoint: navigation shell over the app's sections.

Each page is its own script under pages/ (Streamlit's st.navigation model —
nothing at Python module level survives between pages, only st.session_state
does). Pages are grouped by purpose: Explore (what is available), Solve
(configure and run an algorithm on a problem), Meta-optimization (tune an
algorithm, analyze the result, validate it) and Learn (tutorials).
"""

from pathlib import Path

import streamlit as st

LOGO_DIRECTORY = Path(__file__).parent / "assets" / "logo"

st.set_page_config(
    page_title="Evolver-Studio",
    page_icon=str(LOGO_DIRECTORY / "studio-icon.svg"),
    layout="wide",
)
st.logo(
    str(LOGO_DIRECTORY / "studio-logo.svg"),
    icon_image=str(LOGO_DIRECTORY / "studio-icon.svg"),
    size="large",
)

pg = st.navigation(
    {
        "Explore": [
            st.Page("pages/explore.py", title="Explore", icon="🔍", default=True),
        ],
        "Solve": [
            st.Page("pages/solve.py", title="Run algorithm", icon="▶️"),
        ],
        "Meta-optimization": [
            st.Page("pages/training.py", title="Training", icon="🏋️"),
            st.Page("pages/analysis.py", title="Analysis", icon="📊"),
            st.Page("pages/validation.py", title="Validation", icon="✅"),
        ],
        "Learn": [
            st.Page("pages/tutorials.py", title="Tutorials", icon="🎓"),
        ],
    }
)
pg.run()
