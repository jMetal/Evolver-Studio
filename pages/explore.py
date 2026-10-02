"""Explore: read-only browsing of Evolver's algorithms and their parameter spaces.

Nothing here builds a request or launches a run; which algorithms are actually runnable from this
app is recorded in evolver_studio/catalogue.py and launched from the Training page. Each parameter
space is shown as a table, one row per parameter, which reads at a glance where a nested list
would need a long scroll.
"""

from pathlib import Path

import streamlit as st

from evolver_studio.app_state import require_evolver_jar, warn_if_jar_older_than_catalogue
from evolver_studio.catalogue import BASE_ALGORITHMS, META_ALGORITHMS, MetaAlgorithm
from evolver_studio.parameter_form import render_parameter_space_table
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
        catalogues = [
            (label, filename, parameters)
            for label, filename, parameters in (
                ("Flat", meta.operator_parameter_space_file, meta.flat_parameters),
                ("Tree", meta.tree_operator_parameter_space_file, meta.tree_parameters),
            )
            if filename is not None or parameters
        ]
        tabs = st.tabs([label for label, _, _ in catalogues])
        for tab, (label, filename, parameters) in zip(tabs, catalogues, strict=True):
            with tab:
                if filename is not None:
                    st.caption(f"Operator catalogue ({filename}):")
                    try:
                        text = parameter_space_text(jar, filename)
                    except KeyError:
                        # The catalogue follows Evolver's develop branch, ahead of the release jar.
                        st.warning(f"{filename} is not in this Evolver jar; it needs a newer one.")
                        continue
                    render_parameter_space_table(
                        parse_parameter_space(text), key=f"explorer_meta_{meta.name}_{label}"
                    )
                else:
                    st.write("**Parameters:**", ", ".join(parameters))


st.title("Explore")

jar = require_evolver_jar()
warn_if_jar_older_than_catalogue(jar)

st.subheader("Base algorithms")
selected_name = st.selectbox(
    "Algorithm", [a.name for a in BASE_ALGORITHMS], key="explorer_algorithm"
)
algorithm = next(a for a in BASE_ALGORITHMS if a.name == selected_name)
_render_runnable_badge(algorithm.runnable_today)

encoding = st.selectbox("Encoding", list(algorithm.encodings), key="explorer_encoding")
filename = algorithm.encodings[encoding]
st.caption(f"Parameter space of {algorithm.name} ({encoding}) — {filename}")
render_parameter_space_table(
    parse_parameter_space(parameter_space_text(jar, filename)), key="explorer_base"
)

st.subheader("Meta-optimization algorithms")
for meta in META_ALGORITHMS:
    _render_meta_algorithm_summary(jar, meta)
