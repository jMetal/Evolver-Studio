"""Home: what Evolver-Studio offers, with a card leading to each of its pages."""

from pathlib import Path

import streamlit as st

from evolver_studio.app_state import render_sidebar
from evolver_studio.evolver_client import EVOLVER_VERSION
from evolver_studio.sections import SECTIONS, Page, Section

ICON = Path(__file__).resolve().parent.parent / "assets" / "logo" / "studio-icon.svg"
EVOLVER_URL = "https://github.com/jMetal/Evolver"
EVOLVER_DOCUMENTATION_URL = "https://evolver.readthedocs.io"
CARDS_PER_ROW = 3


def _render_card(section: Section, page: Page) -> None:
    with st.container(border=True, height="stretch"):
        st.caption(section.name.upper())
        st.markdown(f"#### {page.icon} {page.title}")
        if page.available:
            st.badge("Available", icon=":material/check:", color="green")
        else:
            st.badge("Coming soon", icon=":material/construction:", color="gray")
        st.markdown(page.description)
        st.page_link(page.path, label=f"Open {page.title}", icon=":material/arrow_forward:")


jar = render_sidebar()

icon_column, title_column = st.columns([1, 11], vertical_alignment="center")
icon_column.image(str(ICON), width=72)
title_column.title("Evolver-Studio")
st.markdown(
    f"A graphical front end for [Evolver]({EVOLVER_URL}) {EVOLVER_VERSION}, with two uses: "
    "**solving problems** with its configurable multi-objective metaheuristics, and "
    "**meta-optimization**, which finds good configurations of those algorithms automatically "
    "(train, analyze, validate)."
)

if jar is None:
    st.info(
        "**Getting started.** Evolver-Studio runs Evolver, a Java framework (Java 21 or newer is "
        "needed): download its jar from the sidebar. Meanwhile, the **Tutorials** page and "
        f"[Evolver's documentation]({EVOLVER_DOCUMENTATION_URL}) are a good place to start.",
        icon=":material/rocket_launch:",
    )

cards = [(section, page) for section in SECTIONS for page in section.pages]
for start in range(0, len(cards), CARDS_PER_ROW):
    for column, (section, page) in zip(
        st.columns(CARDS_PER_ROW), cards[start : start + CARDS_PER_ROW]
    ):
        with column:
            _render_card(section, page)
