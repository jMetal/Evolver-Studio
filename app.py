"""Evolver-Studio entrypoint: navigation shell over the app's four sections.

Each page is its own script under pages/ (Streamlit's st.navigation model —
nothing at Python module level survives between pages, only st.session_state
does). Ordered to match the natural workflow: understand what's available,
tune it, analyze the result, validate it.
"""

import streamlit as st

st.set_page_config(page_title="Evolver-Studio", layout="wide")

pg = st.navigation(
    [
        st.Page("pages/explore.py", title="Explore", icon="🔍", default=True),
        st.Page("pages/training.py", title="Training", icon="🏋️"),
        st.Page("pages/analysis.py", title="Analysis", icon="📊"),
        st.Page("pages/validation.py", title="Validation", icon="✅"),
    ]
)
pg.run()
