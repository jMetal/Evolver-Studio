"""State shared across pages: the Evolver checkout path and its build control.

Each st.navigation page runs as its own script, so nothing at Python module
level survives between pages — only `st.session_state` does. This sidebar is
rendered once per page (cheap: two widgets) rather than centralized, since
st.navigation has no single "shell" script that wraps every page's body.
"""

from pathlib import Path

import streamlit as st

from evolver_studio.evolver_client import build_jar
from evolver_studio.result import Err

DEFAULT_EVOLVER_HOME = "/Users/ajnebro/Softw/Evolver"


def render_sidebar() -> Path:
    """Render the Evolver checkout path input and build button.

    Returns:
        The Evolver checkout path currently entered in the sidebar.
    """
    evolver_home = Path(
        st.sidebar.text_input("Evolver checkout path", DEFAULT_EVOLVER_HOME, key="evolver_home")
    )
    if st.sidebar.button("Compilar Evolver"):
        build_result = build_jar(evolver_home)
        if isinstance(build_result, Err):
            st.sidebar.error(build_result.message)
        else:
            st.sidebar.success("Jar built successfully.")
    return evolver_home
