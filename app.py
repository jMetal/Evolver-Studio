"""Evolver-Studio entrypoint: navigation shell over the app's sections.

Each page is its own script under pages/ (Streamlit's st.navigation model —
nothing at Python module level survives between pages, only st.session_state
does). The home page leads to the others, which are grouped (see
evolver_studio/sections.py) by purpose: Explore (what is available), Solve
(configure and run an algorithm on a problem), Meta-optimization (tune an
algorithm, analyze the result, validate it) and Learn (tutorials).
"""

from pathlib import Path

import streamlit as st

from evolver_studio.sections import HOME_PAGE, SECTIONS

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
        "": [st.Page(HOME_PAGE, title="Home", icon="🏠", default=True)],
        **{
            section.name: [
                st.Page(page.path, title=page.title, icon=page.icon) for page in section.pages
            ]
            for section in SECTIONS
        },
    }
)
pg.run()
