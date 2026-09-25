"""Explore: read-only browsing of Evolver's algorithms and their parameter spaces.

Nothing here builds a request or launches a run — only NSGA-II/MOEA-D as base
and NSGA-II/SPEA2/SMPSO/AsyncNSGA-II as meta-optimizer are actually
runnable today from this app (see evolver_studio/catalogue.py and the
Training page).
"""

from pathlib import Path

import streamlit as st

from evolver_studio.app_state import require_evolver_jar
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


def _render_meta_algorithm_summary(jar: Path, meta: MetaAlgorithm) -> None:
    """Show one meta-optimizer's encoding support, wiring status, and parameters.

    Prefers the real ParameterSpace YAML backing the algorithm's flat-encoding
    operators (`meta.operator_parameter_space_file`, same format/parser as a
    base algorithm's own parameter space) over the hand-maintained flat name
    list, when one exists — SPEA2/SMPSO hardcode their operators in Java
    instead, so `flat_parameters` is their only option. Same for the tree
    encoding, with `meta.tree_operator_parameter_space_file`/`tree_parameters`.

    Args:
        jar: Path to Evolver's jar, to read the parameter space file when the
            algorithm has one.
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
        if meta.operator_parameter_space_file is not None:
            st.caption(f"Flat operator catalogue ({meta.operator_parameter_space_file}):")
            text = parameter_space_text(jar, meta.operator_parameter_space_file)
            render_parameter_space_readonly(parse_parameter_space(text))
        else:
            st.write("**Parameters (flat):**", ", ".join(meta.flat_parameters))
        if meta.tree_operator_parameter_space_file is not None:
            st.caption(f"Tree operator catalogue ({meta.tree_operator_parameter_space_file}):")
            text = parameter_space_text(jar, meta.tree_operator_parameter_space_file)
            render_parameter_space_readonly(parse_parameter_space(text))
        elif meta.tree_parameters:
            st.write("**Parameters (tree):**", ", ".join(meta.tree_parameters))


st.title("Explore")

jar = require_evolver_jar()

st.subheader("Base algorithms")
selected_name = st.selectbox(
    "Algorithm", [a.name for a in BASE_ALGORITHMS], key="explorer_algorithm"
)
algorithm = next(a for a in BASE_ALGORITHMS if a.name == selected_name)
_render_runnable_badge(algorithm.runnable_today)

encoding = st.selectbox("Encoding", list(algorithm.encodings), key="explorer_encoding")
text = parameter_space_text(jar, algorithm.encodings[encoding])
render_parameter_space_readonly(parse_parameter_space(text))

st.subheader("Meta-optimization algorithms")
for meta in META_ALGORITHMS:
    _render_meta_algorithm_summary(jar, meta)
