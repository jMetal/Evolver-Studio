"""Explore: read-only browsing of Evolver's algorithms and their parameter spaces.

Nothing here builds a request or launches a run — only NSGA-II/MOEA-D as base
and NSGA-II/SPEA2/SMPSO/AsyncNSGA-II as meta-optimizer are actually
runnable today from this app (see evolver_studio/catalogue.py and the
Training page).
"""

import streamlit as st

from evolver_studio.app_state import render_sidebar
from evolver_studio.catalogue import BASE_ALGORITHMS, META_ALGORITHMS, MetaAlgorithm
from evolver_studio.parameter_form import render_parameter_space_readonly
from evolver_studio.parameter_space import parse_parameter_space
from evolver_studio.resource_files import parameter_space_text


def _render_runnable_badge(runnable_today: bool) -> None:
    """Show whether an algorithm can actually be launched today, or only browsed.

    Args:
        runnable_today: Whether cli.training's BaseAlgorithmRegistry supports it.
    """
    if runnable_today:
        st.success("✅ Runnable today from this app.")
    else:
        st.info("🔍 Browsable only for now — Evolver does not expose it for launching a run yet.")


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
        "✅ wired into cli.training"
        if meta.wired_into_cli_runner
        else "🔍 not wired into cli.training"
    )
    with st.expander(f"{meta.name} — {encodings} — {wired}"):
        st.write("**Parameters (flat):**", ", ".join(meta.flat_parameters))
        if meta.tree_parameters:
            st.write("**Parameters (tree):**", ", ".join(meta.tree_parameters))


st.title("Explore")

evolver_home = render_sidebar()

st.subheader("Base algorithms")
selected_name = st.selectbox(
    "Algorithm", [a.name for a in BASE_ALGORITHMS], key="explorer_algorithm"
)
algorithm = next(a for a in BASE_ALGORITHMS if a.name == selected_name)
_render_runnable_badge(algorithm.runnable_today)

encoding = st.selectbox("Encoding", list(algorithm.encodings), key="explorer_encoding")
text = parameter_space_text(evolver_home, algorithm.encodings[encoding])
render_parameter_space_readonly(parse_parameter_space(text))

st.subheader("Meta-optimization algorithms")
for meta in META_ALGORITHMS:
    _render_meta_algorithm_summary(meta)
