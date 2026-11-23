"""Explore › Base algorithms: the parameter space of each configurable algorithm, per encoding.

Read-only: nothing here builds a request or launches a run. Which algorithms are runnable from
this app is recorded in evolver_studio/catalogue.py, and they are launched from the Training page.
Nothing is shown until an algorithm is chosen, so the page opens without a long table nobody asked
for.
"""

import streamlit as st

from evolver_studio.app_state import require_evolver_jar, warn_if_jar_older_than_catalogue
from evolver_studio.catalogue import BASE_ALGORITHMS, CONSTRAINT_HANDLING_PARAMETERS
from evolver_studio.constraint_handling import CONSTRAINT_HANDLING
from evolver_studio.parameter_form import render_parameter_space_table
from evolver_studio.parameter_space import (
    CategoricalParameter,
    ParameterSpec,
    parse_parameter_space,
)
from evolver_studio.problem_catalogue import WITH_CONSTRAINTS, WITHOUT_CONSTRAINTS
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


def _render_constraint_handling(parameters: list[ParameterSpec]) -> None:
    """Say how the algorithm handles constraints: the parameter that chooses it, and its values.

    Args:
        parameters: The parameter space for problems with constraints.
    """
    parameter = next((p for p in parameters if p.name in CONSTRAINT_HANDLING_PARAMETERS), None)
    if not isinstance(parameter, CategoricalParameter):
        return
    values = "\n".join(
        f"- `{choice.value}`: {CONSTRAINT_HANDLING.get(choice.value, '(not described yet)')}"
        for choice in parameter.choices
    )
    st.markdown(f"`{parameter.name}` chooses how the constraints are handled:\n\n{values}")


st.title("Base algorithms")

jar = require_evolver_jar()
warn_if_jar_older_than_catalogue(jar)

selected_name = st.selectbox(
    "Algorithm",
    [a.name for a in BASE_ALGORITHMS],
    index=None,
    placeholder="Choose an algorithm",
    key="explorer_algorithm",
)
if selected_name is None:
    st.caption("Choose an algorithm to see its parameter space.")
    st.stop()

algorithm = next(a for a in BASE_ALGORITHMS if a.name == selected_name)
_render_runnable_badge(algorithm.runnable_today)

encodings = list(algorithm.encodings)
if len(encodings) == 1:
    (encoding,) = encodings
else:
    encoding = st.selectbox("Encoding", encodings, key=f"explorer_encoding_{algorithm.name}")
constrained = False
if algorithm.handles_constraints(encoding):
    constrained = (
        st.radio(
            "Problems",
            (WITHOUT_CONSTRAINTS, WITH_CONSTRAINTS),
            horizontal=True,
            key=f"explorer_constraints_{algorithm.name}_{encoding}",
            help="The space for problems with constraints is the same one plus how the algorithm "
            "handles them. This app uses it whenever a problem has constraints.",
        )
        == WITH_CONSTRAINTS
    )
filename = algorithm.parameter_space_file(encoding, constrained)
parameters = parse_parameter_space(parameter_space_text(jar, filename))
if constrained:
    _render_constraint_handling(parameters)
st.caption(f"Parameter space of {algorithm.name} ({encoding}) — {filename}")
render_parameter_space_table(parameters, key="explorer_base")
