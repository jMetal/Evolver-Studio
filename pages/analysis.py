"""Analysis: study a finished training run and take what it found to where it is used.

Pick a training run kept under cli-runner-runs/ and see how it was run and what it took, how its
meta-objectives converged, how its population evolved, the configurations of its final front and
what they have in common (and where that differs from the algorithm's default configuration). A
configuration goes from here to Validation, to be compared with other algorithms, or to Run
algorithm, to be run on a problem.
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from evolver_studio.app_state import (
    registered_problem_names,
    registered_problems,
    require_evolver_jar,
)
from evolver_studio.configuration import parse_configuration
from evolver_studio.evolver_client import WORKING_DIRECTORY
from evolver_studio.front_analysis import (
    DIFFERENT,
    numeric_values,
    summaries_table,
    summarize_front,
    value_counts,
)
from evolver_studio.monitor_view import (
    MonitorState,
    render_convergence,
    render_indicator_front,
    render_population,
    run_started_at,
)
from evolver_studio.parameter_space import ParameterSpec, parse_parameter_space
from evolver_studio.resource_files import default_configuration_text, parameter_space_text
from evolver_studio.solve_form import CONFIGURATION_VERSION_KEY
from evolver_studio.training_handoff import solve_form_state, validation_form_state
from evolver_studio.training_monitor import format_duration as format_seconds
from evolver_studio.training_runs import FinishedTraining, list_finished_trainings
from evolver_studio.validation import base_algorithm

ALL_CHECKPOINTS = 100_000
SELECTION_KEY = "analysis_training"


def _render_summary(training: FinishedTraining) -> None:
    """Say what was tuned, how the meta-optimizer searched and what it took."""
    problems = ", ".join(training.problems)
    st.markdown(
        f"**{training.algorithm}** ({training.encoding}) tuned on **{problems}**, with "
        f"{', '.join(training.indicators) or 'its meta-objectives'} as meta-objectives."
    )
    columns = st.columns(4)
    meta = (
        f"{training.meta_algorithm} · {training.meta_encoding}" if training.meta_algorithm else "—"
    )
    columns[0].metric("Meta-optimizer", meta)
    population = training.meta_population_size
    columns[1].metric("Meta population", str(population) if population else "default")
    columns[2].metric("Stopped after", training.meta_limit or "—")
    outcome = training.outcome
    took = format_seconds(outcome.wall_clock_seconds) if outcome else "—"
    done = f"{outcome.meta_evaluations:,}" if outcome and outcome.meta_evaluations else "—"
    columns[3].metric("It took", took, help=f"{done} meta-evaluations")
    st.caption(
        f"{len(training.configurations)} configurations on the final front · base population "
        f"{training.population_size or '—'} · "
        f"results in `{training.output_directory}`"
    )


def _monitor_state(training: FinishedTraining) -> MonitorState:
    """The files of a training, read once and kept in the session while it stays chosen."""
    key = f"analysis_files_{training.run_id}"
    if key not in st.session_state:
        output = training.output_directory
        assert output is not None and training.run_dir is not None
        state = MonitorState(
            run_started_at(training.run_id),
            output / "INDICATORS.csv",
            output / "POPULATION_INDICATORS.csv" if training.has_population else None,
            output / "VAR_CONF.txt",
            training.run_dir / "runner.log",
            kept_checkpoints=ALL_CHECKPOINTS,
        )
        state.poll()
        st.session_state[key] = state
    return st.session_state[key]


def _render_configurations(training: FinishedTraining, jar: Path) -> None:
    """List the final front, and offer to validate or run a configuration."""
    configurations = training.configurations
    table = pd.DataFrame(
        [{**c.objectives, "configuration": c.configuration} for c in configurations]
    )
    st.dataframe(table, hide_index=True, width="stretch", height="content")
    index = st.selectbox(
        "Configuration",
        range(len(configurations)),
        format_func=lambda i: f"{i + 1}. {configurations[i].label}",
        key=f"analysis_choice_{training.run_id}",
        help="Each is a different compromise between the meta-objectives; none is better than "
        "another on both. A validation shows how good one really is.",
    )
    chosen = configurations[index]
    st.code(chosen.configuration, language=None, wrap_lines=True)
    columns = st.columns(3)
    columns[0].download_button(
        "Download (.txt)",
        chosen.configuration + "\n",
        file_name=f"configuration_{training.run_id}_{index + 1}.txt",
        key=f"analysis_download_{training.run_id}",
        help="The format a solve request and Validation take: --parameter value pairs.",
    )
    if columns[1].button(
        "Validate it",
        key=f"analysis_validate_{training.run_id}",
        help="Opens Validation with it as the tuned configuration, to compare with the default "
        "of the same algorithm on the problems of this training (add some it never saw).",
    ):
        state = validation_form_state(training, chosen, registered_problems(str(jar)))
        if state is None:
            st.warning("Validation does not offer this algorithm.")
        else:
            st.session_state.update(state)
            st.switch_page("pages/validation.py")
    _render_run_it(training, chosen, jar, columns[2])


def _render_run_it(training: FinishedTraining, chosen, jar: Path, column) -> None:
    problem = st.selectbox(
        "Run it on",
        range(len(training.problems)),
        format_func=lambda i: training.problems[i],
        key=f"analysis_problem_{training.run_id}",
        help="The problem for Run algorithm, with the budget and the reference front it had in "
        "the training.",
    )
    if column.button(
        "Run it",
        key=f"analysis_run_{training.run_id}",
        help="Opens Run algorithm with this configuration, on the problem chosen.",
    ):
        state = solve_form_state(
            training,
            chosen,
            problem,
            registered_problem_names(str(jar)) or [],
            st.session_state.get(CONFIGURATION_VERSION_KEY, 0),
            registered_problems(str(jar)),
        )
        if state is None:
            st.warning("Run algorithm cannot take this configuration or this problem.")
        else:
            st.session_state.update(state)
            st.switch_page("pages/solve.py")


def _space_of(training: FinishedTraining, jar: Path) -> list[ParameterSpec] | None:
    """The parameter space the training searched: its own file, or the algorithm's."""
    file = training.parameter_space_file
    if file is not None and file.is_file():
        return parse_parameter_space(file.read_text())
    algorithm = base_algorithm(training.algorithm)
    if algorithm is None or training.encoding not in algorithm.encodings:
        return None
    return parse_parameter_space(parameter_space_text(jar, algorithm.encodings[training.encoding]))


def _default_values(training: FinishedTraining, jar: Path) -> dict[str, str] | None:
    """The values of the algorithm's default configuration, when it has a single one."""
    algorithm = base_algorithm(training.algorithm)
    defaults = algorithm.default_configurations.get(training.encoding, ()) if algorithm else ()
    if len(defaults) != 1:
        return None
    return parse_configuration(default_configuration_text(jar, defaults[0][1]))


def _render_parameters(training: FinishedTraining, jar: Path) -> None:
    """Show what the configurations of the final front have in common."""
    space = _space_of(training, jar)
    if space is None:
        st.info("The parameter space of this training is not available.")
        return
    configurations = [parse_configuration(c.configuration) for c in training.configurations]
    defaults = _default_values(training, jar)
    summaries = summarize_front(configurations, space, defaults)
    count = len(configurations)
    st.markdown(
        f"What the **{count}** configurations of the final front do with each parameter. Where "
        "they **agree**, the training found that value to matter; where they differ, the "
        "parameter trades one meta-objective for another, or does not matter. "
        + ("With one configuration there is nothing to compare." if count == 1 else "")
    )
    if defaults is None:
        st.caption(
            f"{training.algorithm} has no single default configuration to compare with for "
            f"{training.encoding} problems."
        )
    table = summaries_table(summaries)
    st.dataframe(
        table.style.apply(
            lambda row: (
                ["background-color: rgba(255, 170, 0, 0.18)"] * len(row)
                if row["Vs default"] == DIFFERENT
                else [""] * len(row)
            ),
            axis=1,
        ),
        hide_index=True,
        width="stretch",
        column_config={
            "Agreement": st.column_config.ProgressColumn(
                "Agreement", min_value=0.0, max_value=1.0, format="%.2f"
            )
        },
    )
    changed = [s.name for s in summaries if s.versus_default == DIFFERENT]
    if changed:
        st.caption(f"Highlighted: where the front differs from the default — {', '.join(changed)}.")
    _render_parameter_detail(training, summaries, configurations, defaults)


def _render_parameter_detail(training, summaries, configurations, defaults) -> None:
    """Draw how the configurations spread over one parameter, with the default marked."""
    names = [s.name for s in summaries]
    chosen = st.selectbox("Parameter", names, key=f"analysis_parameter_{training.run_id}")
    summary = next(s for s in summaries if s.name == chosen)
    default = (defaults or {}).get(chosen)
    if summary.kind == "categorical":
        counts = value_counts(configurations, chosen)
        figure = go.Figure(
            go.Bar(
                x=[value for value, _ in counts],
                y=[count for _, count in counts],
                marker_color=[
                    "#e8b04a" if default is not None and value == default else "#1f77b4"
                    for value, _ in counts
                ],
            )
        )
        figure.update_layout(yaxis_title="Configurations", xaxis_title=chosen)
    else:
        values = numeric_values(configurations, chosen)
        figure = go.Figure(go.Box(x=values, name=chosen, boxpoints="all", jitter=0.4, pointpos=0))
        if default is not None:
            try:
                figure.add_vline(x=float(default), line_dash="dash", line_color="#e8b04a")
            except ValueError:
                pass
        figure.update_layout(xaxis_title=chosen, showlegend=False)
    figure.update_layout(height=300, margin={"t": 20})
    st.plotly_chart(figure, width="stretch", key=f"analysis_detail_{training.run_id}_{chosen}")
    if default is not None:
        st.caption("The default configuration's value is marked in orange.")


st.title("Analysis")

jar = require_evolver_jar()

trainings = list_finished_trainings(WORKING_DIRECTORY)
if not trainings:
    st.info(
        "There is no finished training run to analyze: the ones you run in Training are kept "
        "and listed here."
    )
    st.page_link("pages/training.py", label="Go to Training", icon="🏋️")
    st.stop()

training = st.selectbox("Training run", trainings, format_func=lambda t: t.label, key=SELECTION_KEY)
_render_summary(training)
state = _monitor_state(training)
names = ["Convergence", "Front", *(["Population"] if training.has_population else [])]
names += ["Configurations", "Parameters"]
tabs = iter(st.tabs(names))
key = f"analysis_{training.run_id}"
with next(tabs):
    render_convergence(state, f"{key}_convergence")
with next(tabs):
    render_indicator_front(training.output_directory / "INDICATORS.csv", training.run_id)
if training.has_population:
    with next(tabs):
        render_population(state, f"{key}_population")
with next(tabs):
    _render_configurations(training, jar)
with next(tabs):
    _render_parameters(training, jar)
