"""Explore › Meta-optimizers: the encodings each meta-optimizer supports and its operators.

Read-only, like the base algorithms page, and empty until a meta-optimizer is chosen. Its operators
are shown from the real ParameterSpace YAML backing them (`operator_parameter_space_file` for the
flat encoding, `tree_operator_parameter_space_file` for the tree one) when there is one; SPEA2 and
SMPSO hardcode their operators in Java instead, so the hand-maintained name list is shown for them.
"""

from pathlib import Path

import streamlit as st

from evolver_studio.app_state import require_evolver_jar, warn_if_jar_older_than_catalogue
from evolver_studio.catalogue import META_ALGORITHMS, MetaAlgorithm
from evolver_studio.parameter_form import render_parameter_space_table
from evolver_studio.parameter_space import parse_parameter_space
from evolver_studio.resource_files import parameter_space_text


def _render_status(meta: MetaAlgorithm) -> None:
    """Show which encodings a meta-optimizer supports and whether the app can launch it.

    Args:
        meta: The meta-optimizer to describe.
    """
    encodings = " and ".join(
        encoding
        for encoding, supported in (("flat", meta.supports_flat), ("tree", meta.supports_tree))
        if supported
    )
    if meta.wired_into_cli_runner:
        st.success(f"✅ Runnable today from this app, with the {encodings} encoding.")
    else:
        st.info(
            f"🔍 Supports the {encodings} encoding, but Evolver does not expose it for launching "
            "a run yet."
        )


def _render_operator_catalogues(jar: Path, meta: MetaAlgorithm) -> None:
    """Show a meta-optimizer's operator catalogues, one tab per encoding.

    Args:
        jar: Path to Evolver's jar, to read the parameter space files.
        meta: The meta-optimizer whose operators to show.
    """
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
            if filename is None:
                st.write("**Parameters:**", ", ".join(parameters))
                continue
            st.caption(f"Operator catalogue ({filename}):")
            try:
                text = parameter_space_text(jar, filename)
            except KeyError:
                st.warning(f"{filename} is not in this Evolver jar; it needs a newer one.")
                continue
            render_parameter_space_table(
                parse_parameter_space(text), key=f"explorer_meta_{meta.name}_{label}"
            )


st.title("Meta-optimizers")

jar = require_evolver_jar()
warn_if_jar_older_than_catalogue(jar)

selected_name = st.selectbox(
    "Meta-optimizer",
    [m.name for m in META_ALGORITHMS],
    index=None,
    placeholder="Choose a meta-optimizer",
    key="explorer_meta_algorithm",
)
if selected_name is None:
    st.caption("Choose a meta-optimizer to see its encodings and operators.")
    st.stop()

meta = next(m for m in META_ALGORITHMS if m.name == selected_name)
_render_status(meta)
_render_operator_catalogues(jar, meta)
