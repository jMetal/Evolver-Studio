"""Evolver-Studio entrypoint: navigation shell over the app's sections.

Each page is its own script under pages/ (Streamlit's st.navigation model —
nothing at Python module level survives between pages, only st.session_state
does). Pages are grouped by purpose: Explore (what is available), Solve
(configure and run an algorithm on a problem), Meta-optimization (tune an
algorithm, analyze the result, validate it) and Learn (tutorials).
"""

import streamlit as st

st.set_page_config(page_title="Evolver-Studio", layout="wide")

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
