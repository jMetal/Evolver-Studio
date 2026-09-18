"""Explorar: read-only browsing of Evolver's algorithms and their parameter spaces.

Nothing here builds a request or launches a run — only NSGA-II/MOEA-D as base
and NSGA-II/SPEA2/SMPSO/AsyncNSGA-II as meta-optimizer are actually
runnable today from this app (see evolver_studio/catalogue.py and the
Entrenamiento page).
"""

import streamlit as st

from evolver_studio.app_state import render_sidebar
from evolver_studio.catalogue import BASE_ALGORITHMS, META_ALGORITHMS, MetaAlgorithm
from evolver_studio.parameter_form import render_parameter_form
from evolver_studio.parameter_space import parse_parameter_space
from evolver_studio.resource_files import parameter_space_text


def _render_runnable_badge(runnable_today: bool) -> None:
    """Show whether an algorithm can actually be launched today, or only browsed.

    Args:
        runnable_today: Whether cli.training's BaseAlgorithmRegistry supports it.
    """
    if runnable_today:
        st.success("✅ Ejecutable hoy desde esta app.")
    else:
        st.info("🔍 Solo explorable por ahora — Evolver aún no lo expone para lanzar un run.")


def _render_meta_algorithm_summary(meta: MetaAlgorithm) -> None:
    """Show one meta-optimizer's encoding support, wiring status, and parameters.

    Args:
        meta: The meta-optimizer to summarize.
    """
    encodings = ", ".join(
        encoding
        for encoding, supported in (("flat", meta.supports_flat), ("tree", meta.supports_tree))
        if supported
    )
    wired = (
        "✅ conectado a cli.training"
        if meta.wired_into_cli_runner
        else "🔍 no conectado a cli.training"
    )
    with st.expander(f"{meta.name} — {encodings} — {wired}"):
        st.write("**Parámetros (flat):**", ", ".join(meta.flat_parameters))
        if meta.tree_parameters:
            st.write("**Parámetros (tree):**", ", ".join(meta.tree_parameters))


st.title("Explorar")

evolver_home = render_sidebar()

st.subheader("Algoritmos base")
selected_name = st.selectbox(
    "Algoritmo", [a.name for a in BASE_ALGORITHMS], key="explorer_algorithm"
)
algorithm = next(a for a in BASE_ALGORITHMS if a.name == selected_name)
_render_runnable_badge(algorithm.runnable_today)

encoding = st.selectbox("Codificación", list(algorithm.encodings), key="explorer_encoding")
text = parameter_space_text(evolver_home, algorithm.encodings[encoding])
render_parameter_form(parse_parameter_space(text), f"explorer_{selected_name}_{encoding}")

st.subheader("Algoritmos de meta-optimización")
for meta in META_ALGORITHMS:
    _render_meta_algorithm_summary(meta)
