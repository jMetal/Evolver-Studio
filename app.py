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
        st.Page("pages/explorar.py", title="Explorar", icon="🔍", default=True),
        st.Page("pages/entrenamiento.py", title="Entrenamiento", icon="🏋️"),
        st.Page("pages/analisis.py", title="Análisis", icon="📊"),
        st.Page("pages/validacion.py", title="Validación", icon="✅"),
    ]
)
pg.run()
