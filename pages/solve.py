"""Run algorithm: run one of Evolver's configurable algorithms on a problem.

The page builds a solve request (a problem, an algorithm with a configuration, a budget), launches
Evolver's cli.solving as a detached subprocess, and shows the fronts and the indicators of the runs.
The configuration starts from the algorithm's default one and can be adjusted within its parameter
space. Every run is kept under solve-runs/, so a past one can be reopened.
"""

import datetime as dt
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

from evolver_studio.app_state import (
    registered_problem_names,
    require_evolver_jar,
    warn_if_jar_older_than_catalogue,
)
from evolver_studio.catalogue import (
    BASE_ALGORITHMS,
    QUALITY_INDICATORS,
    BaseAlgorithm,
    is_at_least,
)
from evolver_studio.configuration import (
    complete_values,
    configuration_string,
    modified_values,
    parse_configuration,
    starting_values,
    values_outside_the_space,
)
from evolver_studio.evolver_client import (
    EVOLVER_VERSION,
    WORKING_DIRECTORY,
    RunStatus,
    cancel_training,
    read_pid,
    read_status,
    start_solve,
    write_pid_file,
)
from evolver_studio.parameter_form import render_configuration_form
from evolver_studio.parameter_space import parse_parameter_space
from evolver_studio.problems import (
    default_reference_front,
    reference_front_candidates,
    reference_front_dimension,
)
from evolver_studio.progress import estimate_remaining_seconds, format_duration
from evolver_studio.resource_files import (
    default_configuration_text,
    jar_evolver_version,
    parameter_space_text,
)
from evolver_studio.runs import (
    SOLVE_RUNS_DIRECTORY_NAME,
    RunPhase,
    find_run_in_progress,
    mark_cancelled,
    run_phase,
)
from evolver_studio.solve_figures import build_front_figure, build_live_front_figure
from evolver_studio.solve_form import (
    ALGORITHM_KEY,
    CONFIGURATION_VERSION_KEY,
    EVALUATIONS_KEY,
    FIX_SEED_KEY,
    LOADED_CONFIGURATION_KEY,
    NO_REFERENCE_FRONT,
    POPULATION_CHOICE_KEY,
    POPULATION_KEY,
    PROBLEM_KEY,
    RUNS_KEY,
    SEED_KEY,
    WEIGHT_VECTORS_KEY,
    encoding_key,
    form_state_from_request,
    indicators_key,
    reference_front_key,
)
from evolver_studio.solve_request import SolveRequest, solve_request_to_yaml
from evolver_studio.solve_results import (
    SolveRunInfo,
    filter_by_objectives,
    indicator_summary,
    is_permutation,
    list_solve_runs,
    objective_columns,
    read_current_front,
    read_front,
    read_indicators,
    read_request,
    read_run_fronts,
    read_solutions,
    zip_fronts,
)
from evolver_studio.weight_vectors import available_population_sizes

RUNS_DIRECTORY = WORKING_DIRECTORY / SOLVE_RUNS_DIRECTORY_NAME
DEFAULT_INDICATORS = ("Epsilon", "NormalizedHypervolume")
DEFAULT_WEIGHT_VECTORS_DIRECTORY = "resources/weightVectors"
DEFAULT_POPULATION_SIZE = 100
LIVE_FRAGMENT_RUN_EVERY_SECONDS = 1
LOG_LINES_SHOWN = 15
# The evaluations between two updates of the progress that the slider offers, and the one it starts
# at. Evolver reports in steps of an algorithm's offspring population (usually 100): updating more
# often than that costs time and shows nothing more, except for a steady-state algorithm.
PROGRESS_FREQUENCIES = (1, 10, 100, 500, 1000, 5000, 10000, 50000)
DEFAULT_PROGRESS_FREQUENCY = 500
SLOW_PROGRESS_FREQUENCY_BELOW = 100
DEFAULT_FRONT_FREQUENCY = 1000
TRACKING_SILENT = "Silent"
TRACKING_PROGRESS = "Progress bar"
TRACKING_FRONT = "Live front"
TRACKING_MODES = (TRACKING_SILENT, TRACKING_PROGRESS, TRACKING_FRONT)
# The Evolver release from which a run reports its progress while it runs.
PROGRESS_WITHIN_A_RUN_SINCE = "2.3"


def _render_problem() -> tuple[str, str | None] | None:
    """Section 1: the problem and its reference front.

    Returns:
        The problem's name and its reference front file (None for no indicators), or None while
        either is not chosen yet.
    """
    st.subheader("1. Problem")
    names = registered_problem_names(str(jar))
    if names is None:
        st.warning("Could not list the problems (is Java installed?).")
        return None
    problem = st.selectbox(
        "Problem", names, index=None, placeholder="Choose a problem", key=PROBLEM_KEY
    )
    if problem is None:
        return None
    candidates = reference_front_candidates(problem)
    if not candidates:
        st.info("This problem has no reference front among the resources: no indicators.")
        return problem, None
    choice = st.selectbox(
        "Reference front",
        [*candidates, NO_REFERENCE_FRONT],
        index=_default_front_index(candidates),
        placeholder="Choose the reference front",
        key=reference_front_key(problem),
        help="The indicators compare each front with it. Fronts with a dimension suffix (3D) "
        "belong to the problem with that number of objectives: it must match the problem's.",
    )
    if choice is None:
        return None
    return problem, None if choice == NO_REFERENCE_FRONT else choice


def _default_front_index(candidates: list[str]) -> int | None:
    default = default_reference_front(candidates)
    return candidates.index(default) if default is not None else None


def _render_algorithm() -> tuple[BaseAlgorithm, str, dict[str, str] | None] | None:
    """Section 2: the algorithm, its encoding and what it needs besides.

    Returns:
        The algorithm, the encoding and the extra configuration (None if it needs none), or None
        while no algorithm is chosen.
    """
    st.subheader("2. Algorithm")
    runnable = [algorithm for algorithm in BASE_ALGORITHMS if algorithm.runnable_today]
    name = st.selectbox(
        "Algorithm",
        [algorithm.name for algorithm in runnable],
        index=None,
        placeholder="Choose an algorithm",
        key=ALGORITHM_KEY,
    )
    if name is None:
        return None
    algorithm = next(algorithm for algorithm in runnable if algorithm.name == name)
    encodings = algorithm.runnable_encodings
    if len(encodings) == 1:
        (encoding,) = encodings
    else:
        encoding = st.selectbox("Encoding", encodings, key=encoding_key(algorithm.name))
    extra_config = None
    if "weightVectorFilesDirectory" in algorithm.required_extra_config_keys:
        directory = st.text_input(
            "Weight vector files directory",
            DEFAULT_WEIGHT_VECTORS_DIRECTORY,
            help="The population size must match one of its files (W<objectives>D_<size>.dat).",
            key=WEIGHT_VECTORS_KEY,
        )
        extra_config = {"weightVectorFilesDirectory": directory}
    return algorithm, encoding, extra_config


def _render_configuration(algorithm: BaseAlgorithm, encoding: str) -> str:
    """Section 3: the default configuration, adjustable within the parameter space.

    Args:
        algorithm: The chosen algorithm.
        encoding: The chosen encoding.

    Returns:
        The configuration string, "--parameter value ...".
    """
    st.subheader("3. Configuration")
    parameters = parse_parameter_space(parameter_space_text(jar, algorithm.encodings[encoding]))
    defaults = algorithm.default_configurations.get(encoding, ())
    if defaults:
        labels = [label for label, _ in defaults]
        label = (
            st.radio(
                "Default configuration",
                labels,
                horizontal=True,
                key=f"solve_default_{algorithm.name}_{encoding}",
            )
            if len(labels) > 1
            else labels[0]
        )
        filename = dict(defaults)[label]
        reference = parse_configuration(default_configuration_text(jar, filename))
        st.caption(f"Starting from `{filename}`; adjust the parameters below if you want.")
    else:
        label = "starting values"
        reference = starting_values(parameters)
        st.caption(
            "Evolver has no default configuration for this algorithm and encoding: starting from "
            "the first value of each choice and the middle of each range of the parameter space."
        )
    loaded = st.session_state.get(LOADED_CONFIGURATION_KEY)
    if loaded and (loaded["algorithm"], loaded["encoding"]) == (algorithm.name, encoding):
        st.caption(
            "Starting from the configuration of the run you chose; the marks show what "
            "differs from the default one."
        )
        values = complete_values(parameters, loaded["values"])
    else:
        values = complete_values(parameters, reference)
    version = st.session_state.setdefault(CONFIGURATION_VERSION_KEY, 0)
    key_prefix = f"solve_configuration_{algorithm.name}_{encoding}_{label}_{version}"
    with st.expander("Adjust the parameters"):
        if st.button("Reset to the default configuration"):
            st.session_state[CONFIGURATION_VERSION_KEY] = version + 1
            st.session_state.pop(LOADED_CONFIGURATION_KEY, None)
            st.rerun()
        edited = render_configuration_form(parameters, values, reference, key_prefix)
    outside = values_outside_the_space(parameters, edited)
    if outside:
        st.warning("Outside the parameter space: " + ", ".join(f"`{name}`" for name in outside))
    changed = modified_values(parameters, edited, reference)
    st.caption(f"{len(changed)} parameter(s) changed from the starting configuration.")
    configuration = configuration_string(parameters, edited)
    st.code(configuration, language=None, wrap_lines=True)
    return configuration


def _render_population_size(column, population_sizes: list[int] | None) -> int:
    """Choose the population size: any, or one a weight vector file exists for.

    Args:
        column: The column to draw it in.
        population_sizes: The sizes the weight vector files allow (MOEA/D and RVEA need a file
            that matches), an empty list when there is none for the problem's objectives, or None
            when the algorithm does not read weight vectors or the objectives are not known.

    Returns:
        The population size.
    """
    if population_sizes:
        default = DEFAULT_POPULATION_SIZE
        return column.selectbox(
            "Population size",
            population_sizes,
            index=population_sizes.index(default) if default in population_sizes else 0,
            help="These sizes have a weight vector file for the problem's number of objectives.",
            key=POPULATION_CHOICE_KEY,
        )
    population_size = column.number_input(
        "Population size", 2, value=DEFAULT_POPULATION_SIZE, key=POPULATION_KEY
    )
    if population_sizes is not None:
        column.warning("No weight vector file for this number of objectives in that directory.")
    return int(population_size)


@dataclass(slots=True, frozen=True)
class Tracking:
    """What section 4 chooses about following a run while it runs.

    Attributes:
        status_frequency: Every how many evaluations the progress is updated, or None to update it
            only when each independent run ends.
        front_frequency: Every how many evaluations the current front is written, or None for none.
        write_population: Whether that file holds the whole population, not only the
            non-dominated solutions.
    """

    status_frequency: int | None = None
    front_frequency: int | None = None
    write_population: bool = False


@dataclass(slots=True, frozen=True)
class Budget:
    """What section 4 chooses.

    Attributes:
        population_size: The population size.
        max_evaluations: The evaluations of each run.
        runs: The number of independent runs.
        seed: The seed of the first run, or None to draw one.
        indicators: The quality indicators to compute.
        tracking: How the run is followed while it is in progress.
    """

    population_size: int
    max_evaluations: int
    runs: int
    seed: int | None
    indicators: list[str]
    tracking: Tracking


def _frequency_warning(mode: str, frequency: int) -> str | None:
    """Say what updating this often costs, from what was measured on NSGA-II (see Evolver's
    docs/utilities/cli_tools.rst), or None when it costs little. Showing the front is expected to
    slow the run down, so only an excessive frequency is warned about."""
    if mode == TRACKING_FRONT:
        if frequency < SLOW_PROGRESS_FREQUENCY_BELOW:
            return (
                f"Writing the front every {frequency} evaluation(s) can make the run several times "
                "slower (more than ten times for an algorithm that reports every evaluation)."
            )
        return None
    if frequency < SLOW_PROGRESS_FREQUENCY_BELOW:
        return (
            f"Updating every {frequency} evaluation(s) can slow the run down a lot: up to about "
            "twice as long in the cases measured. From 100 evaluations on, the cost is a few "
            "percent."
        )
    return None


def _render_tracking(max_evaluations: int, runs: int) -> Tracking:
    """Choose how the run is followed while it is in progress, and how often it is refreshed.

    Args:
        max_evaluations: The evaluations of each run.
        runs: The number of independent runs.

    Returns:
        What was chosen: nothing (silent, or an Evolver that cannot do more), the progress, or the
        progress and the front.
    """
    version = jar_evolver_version(jar)
    supported = version is None or is_at_least(version, PROGRESS_WITHIN_A_RUN_SINCE)
    modes = TRACKING_MODES if supported else (TRACKING_SILENT,)
    mode = st.radio(
        "While the run is in progress",
        modes,
        index=modes.index(TRACKING_PROGRESS) if supported else 0,
        horizontal=True,
        disabled=not supported,
        help="Silent updates the progress only when each independent run ends. Progress bar "
        "updates it every N evaluations. Live front also plots the front as it evolves. A run "
        "that lasts a second or two (MOEA/D on a small budget, for instance) ends before there is "
        "anything to follow: the live view is for the longer ones.",
        key="solve_tracking",
    )
    if not supported:
        st.caption(
            f"Evolver {version} updates the progress only when each run ends; Evolver "
            f"{PROGRESS_WITHIN_A_RUN_SINCE} or later does it while the run is in progress."
        )
        return Tracking()
    if mode == TRACKING_SILENT:
        st.caption("The progress changes only when each independent run ends.")
        return Tracking()
    frequency = st.select_slider(
        "Update every N evaluations",
        options=PROGRESS_FREQUENCIES,
        value=DEFAULT_FRONT_FREQUENCY if mode == TRACKING_FRONT else DEFAULT_PROGRESS_FREQUENCY,
        help="How often it is refreshed. Each update costs time: the more often, the longer the "
        "run takes, and plotting the front costs much more than the progress bar.",
        key=f"solve_update_every_{mode}",
    )
    frequency = min(frequency, max_evaluations)
    st.caption(f"About {max(runs * max_evaluations // frequency, 1)} updates in all.")
    warning = _frequency_warning(mode, frequency)
    if warning is not None:
        st.warning(warning)
    if mode == TRACKING_PROGRESS:
        return Tracking(status_frequency=frequency)
    whole_population = st.checkbox(
        "Show the dominated solutions too",
        help="Plots the whole population, the non-dominated solutions in color and the dominated "
        "ones in grey. It is a larger file to write each time.",
        key="solve_whole_population",
    )
    return Tracking(
        status_frequency=frequency, front_frequency=frequency, write_population=whole_population
    )


def _render_budget(has_reference_front: bool, population_sizes: list[int] | None) -> Budget:
    """Section 4: the population, the budget, the runs, the seed, the indicators and the progress.

    Args:
        has_reference_front: Whether the problem has a reference front (indicators need one).
        population_sizes: The population sizes the weight vector files allow, if the algorithm
            reads them (see `_render_population_size`).

    Returns:
        What was chosen.
    """
    st.subheader("4. Budget")
    columns = st.columns(4)
    population_size = _render_population_size(columns[0], population_sizes)
    max_evaluations = columns[1].number_input(
        "Evaluations per run", 100, value=25000, step=1000, key=EVALUATIONS_KEY
    )
    runs = columns[2].number_input("Independent runs", 1, value=1, key=RUNS_KEY)
    with columns[3]:
        fix_seed = st.checkbox("Fix the seed", key=FIX_SEED_KEY)
        seed = st.number_input("Seed", 0, value=1, disabled=not fix_seed, key=SEED_KEY)
    indicators = st.multiselect(
        "Quality indicators",
        [indicator.registry_name for indicator in QUALITY_INDICATORS],
        default=list(DEFAULT_INDICATORS) if has_reference_front else [],
        disabled=not has_reference_front,
        help="Computed on each run's front, normalized with the reference front. Run `i` uses "
        "the seed + i - 1.",
        key=indicators_key(has_reference_front),
    )
    tracking = _render_tracking(int(max_evaluations), int(runs))
    return Budget(
        population_size=int(population_size),
        max_evaluations=int(max_evaluations),
        runs=int(runs),
        seed=int(seed) if fix_seed else None,
        indicators=indicators,
        tracking=tracking,
    )


def _launch(request_for: Callable[[str], SolveRequest]) -> None:
    """Write the request of a new run, start it and remember it."""
    run_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = RUNS_DIRECTORY / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    request = request_for(f"{SOLVE_RUNS_DIRECTORY_NAME}/{run_id}/output")
    (run_dir / "request.yaml").write_text(solve_request_to_yaml(request))
    process = start_solve(
        WORKING_DIRECTORY,
        jar,
        run_dir / "request.yaml",
        run_dir / "status.yaml",
        run_dir / "runner.log",
    )
    write_pid_file(run_dir / "pid.txt", process.pid)
    st.session_state.pop("solve_history", None)


def _weight_vector_population_sizes(
    extra_config: dict[str, str] | None, reference_front: str | None
) -> list[int] | None:
    """The population sizes the algorithm's weight vector files allow, when it reads them.

    Args:
        extra_config: The algorithm's extra configuration (None if it reads no weight vectors).
        reference_front: The chosen reference front, whose columns give the objectives.

    Returns:
        The sizes (an empty list if there is no file for those objectives), or None when the
        algorithm reads no weight vectors or the number of objectives is not known.
    """
    if extra_config is None or reference_front is None:
        return None
    objectives = reference_front_dimension(WORKING_DIRECTORY / reference_front)
    if objectives is None:
        return None
    directory = WORKING_DIRECTORY / extra_config["weightVectorFilesDirectory"]
    return available_population_sizes(directory, objectives)


def _render_form(can_run: bool) -> None:
    """The five sections that build and launch a run, each shown once the previous is chosen.

    The form stays on screen while a run is in progress (unmounting its widgets would lose what
    the user chose), with the Run button disabled.

    Args:
        can_run: Whether a run can be started: none is in progress.
    """
    problem = _render_problem()
    if problem is None:
        return
    problem_name, reference_front = problem
    algorithm_choice = _render_algorithm()
    if algorithm_choice is None:
        return
    algorithm, encoding, extra_config = algorithm_choice
    configuration = _render_configuration(algorithm, encoding)
    budget = _render_budget(
        reference_front is not None, _weight_vector_population_sizes(extra_config, reference_front)
    )

    def request_for(output_directory: str) -> SolveRequest:
        return SolveRequest(
            algorithm_name=algorithm.registry_name,
            encoding=encoding,
            population_size=budget.population_size,
            yaml_parameter_space_file=algorithm.encodings[encoding],
            extra_config=extra_config,
            configuration=configuration,
            problem=problem_name,
            reference_front_file_name=reference_front,
            max_evaluations=budget.max_evaluations,
            number_of_independent_runs=budget.runs,
            seed=budget.seed,
            indicator_names=budget.indicators,
            status_frequency=budget.tracking.status_frequency,
            front_frequency=budget.tracking.front_frequency,
            write_population=budget.tracking.write_population,
            output_directory=output_directory,
        )

    errors = request_for("").validation_errors()
    for error in errors:
        st.warning(error)
    if st.button("Run", type="primary", disabled=bool(errors) or not can_run):
        _launch(request_for)
        st.rerun()


def _cancel(run_dir: Path) -> None:
    """Stop a run's process and mark it cancelled."""
    pid = read_pid(run_dir / "pid.txt")
    if pid is not None:
        cancel_training(pid)
    mark_cancelled(run_dir)


def _progress_text(run_dir: Path, status: RunStatus) -> str:
    """Write a run's progress: the evaluations, the time elapsed and the time left."""
    text = f"{status.evaluations_done}/{status.max_evaluations} evaluations"
    try:
        started = dt.datetime.strptime(run_dir.name, "%Y%m%d-%H%M%S")
    except ValueError:
        return text
    elapsed = (dt.datetime.now() - started).total_seconds()
    text += f" · {format_duration(elapsed)} elapsed"
    remaining = estimate_remaining_seconds(elapsed, status.evaluations_done, status.max_evaluations)
    if remaining is not None:
        text += f" · about {format_duration(remaining)} left"
    return text


@st.cache_data(show_spinner=False)
def _reference_front(file_name: str) -> pd.DataFrame:
    """Read a problem's reference front, once: it can be large and the live chart redraws often."""
    return read_front(WORKING_DIRECTORY / file_name)


def _render_live_front(run_dir: Path, request: dict) -> None:
    """Plot the front of the run in progress, as Evolver last wrote it."""
    solutions = read_current_front(WORKING_DIRECTORY / request["outputDirectory"])
    if solutions is None:
        st.caption("Waiting for the first front…")
        return
    reference_file = request.get("referenceFrontFileName")
    reference = _reference_front(reference_file) if reference_file else None
    st.plotly_chart(
        build_live_front_figure(solutions, reference),
        width="stretch",
        key=f"solve_live_front_{run_dir.name}",
    )


def _render_run_in_progress(run_dir: Path) -> None:
    """Show the progress of the run in progress, and move on to its results when it ends."""
    st.info(f"Running ({run_dir.name}).")
    request = read_request(run_dir) or {}
    if "statusFrequency" not in request:
        st.caption("This run reports its progress only when each of its independent runs ends.")
    if st.button("Cancel"):
        _cancel(run_dir)
        st.rerun()

    @st.fragment(run_every=LIVE_FRAGMENT_RUN_EVERY_SECONDS, key=f"solve_poll_{run_dir.name}")
    def _poll() -> None:
        status = read_status(run_dir / "status.yaml")
        if status is not None:
            fraction = min(status.evaluations_done / max(status.max_evaluations, 1), 1.0)
            st.progress(fraction, text=_progress_text(run_dir, status))
        else:
            st.progress(0.0, text="Starting…")
        if "frontFrequency" in request:
            _render_live_front(run_dir, request)
        if run_phase(run_dir) not in (RunPhase.STARTING, RunPhase.RUNNING):
            st.session_state["solve_history"] = run_dir.name
            st.rerun()

    _poll()


def _render_results(run: SolveRunInfo) -> None:
    """Show a run's outcome: its fronts and indicators, or why it has none."""
    phase = run_phase(run.run_dir)
    if phase == RunPhase.CANCELLED:
        st.info("This run was cancelled.")
        return
    if phase in (RunPhase.FAILED, RunPhase.LOST):
        status = read_status(run.run_dir / "status.yaml")
        st.error(
            status.error_message
            if status is not None and status.error_message
            else "The run ended without writing its results."
        )
        _render_runner_log(run.run_dir / "runner.log")
        return
    if phase != RunPhase.FINISHED:
        st.info("This run is still in progress.")
        return
    tabs = st.tabs(["Front", "Indicators", "Solutions", "Details"])
    with tabs[0]:
        _render_front(run)
    with tabs[1]:
        _render_indicators(run)
    with tabs[2]:
        _render_solutions(run)
    with tabs[3]:
        _render_details(run)


def _render_runner_log(log_file: Path) -> None:
    if log_file.is_file() and (text := log_file.read_text().strip()):
        with st.expander("Runner output"):
            st.code("\n".join(text.splitlines()[-LOG_LINES_SHOWN:]), language=None)


def _render_front(run: SolveRunInfo) -> None:
    fronts = read_run_fronts(run.output_directory)
    chosen = st.multiselect(
        "Runs", list(fronts), default=list(fronts), format_func=lambda number: f"Run {number}"
    )
    reference_file = run.request.get("referenceFrontFileName")
    reference = read_front(WORKING_DIRECTORY / reference_file) if reference_file else None
    st.plotly_chart(
        build_front_figure({number: fronts[number] for number in chosen}, reference),
        width="stretch",
        key=f"solve_front_{run.run_id}",
    )


def _render_solutions(run: SolveRunInfo) -> None:
    """Show the solutions of one run, with their objectives and variables.

    The table can be filtered by a range for each objective, sorted by any column, downloaded, and
    a row can be selected to see that solution's decision variables.
    """
    numbers = list(read_run_fronts(run.output_directory))
    number = st.selectbox(
        "Run",
        numbers,
        format_func=lambda n: f"Run {n}",
        key=f"solve_solutions_run_{run.run_id}",
    )
    solutions = read_solutions(run.output_directory, number)
    shown = filter_by_objectives(solutions, _objective_ranges(run, number, solutions))
    st.caption(f"{len(shown)} of {len(solutions)} solutions.")
    selection = st.dataframe(
        shown,
        width="stretch",
        on_select="rerun",
        selection_mode="single-row",
        key=f"solve_solutions_table_{run.run_id}_{number}",
    )
    if selection.selection.rows:
        position = selection.selection.rows[0]
        _render_solution(shown.iloc[position], shown.index[position])
    st.download_button(
        "Download this table (CSV)",
        shown.to_csv(index_label="solution"),
        file_name=f"{run.run_id}-run-{number}-solutions.csv",
        mime="text/csv",
        key=f"solve_solutions_download_{run.run_id}_{number}",
    )


def _objective_ranges(
    run: SolveRunInfo, number: int, solutions: pd.DataFrame
) -> dict[str, tuple[float, float]]:
    """Let the user restrict each objective to a range; only the restricted ones are returned."""
    ranges: dict[str, tuple[float, float]] = {}
    objectives = objective_columns(solutions)
    with st.expander("Filter by objective"):
        columns = st.columns(min(len(objectives), 4))
        for index, objective in enumerate(objectives):
            lower, upper = float(solutions[objective].min()), float(solutions[objective].max())
            if lower == upper:
                continue
            ranges[objective] = columns[index % len(columns)].slider(
                objective,
                lower,
                upper,
                (lower, upper),
                format="%.4g",
                key=f"solve_filter_{run.run_id}_{number}_{objective}",
            )
    return ranges


def _render_solution(solution: pd.Series, identifier: int) -> None:
    """Show one solution: its objectives, and its decision variables."""
    st.markdown(f"**Solution {identifier}**")
    st.caption(
        ", ".join(f"{name} = {solution[name]:.6g}" for name in solution.index if name[0] == "f")
    )
    variables = solution[[name for name in solution.index if name[0] == "x"]]
    if variables.empty:
        return
    if is_permutation(variables):
        st.caption("A permutation of 0 to n-1:")
        st.code(" ".join(str(int(value)) for value in variables), language=None, wrap_lines=True)
    else:
        st.caption("Decision variables:")
        st.bar_chart(variables)


def _render_indicators(run: SolveRunInfo) -> None:
    if not run.request.get("indicatorNames"):
        st.info("This run computed no indicators.")
        return
    indicators = read_indicators(run.output_directory)
    st.dataframe(indicators, hide_index=True, width="stretch")
    if len(indicators) > 1:
        st.markdown("**Over the runs**")
        st.dataframe(indicator_summary(indicators), width="stretch")


def _load_run(run: SolveRunInfo) -> None:
    """Set the form to the settings of a past run (a button's callback, so it runs before the
    form's widgets are created and may set their values)."""
    names = registered_problem_names(str(jar)) or []
    state = form_state_from_request(
        run.request,
        BASE_ALGORITHMS,
        names,
        st.session_state.get(CONFIGURATION_VERSION_KEY, 0),
    )
    if state is None:
        st.session_state["solve_load_error"] = run.run_id
        return
    st.session_state.pop("solve_load_error", None)
    st.session_state.update(state)


def _render_details(run: SolveRunInfo) -> None:
    st.write(f"Results folder: `{run.output_directory}`")
    st.markdown("**Configuration**")
    st.code(run.request["configuration"], language=None, wrap_lines=True)
    with st.expander("METADATA.txt"):
        st.text((run.output_directory / "METADATA.txt").read_text())
    st.button(
        "Use this configuration for a new run",
        on_click=_load_run,
        args=(run,),
        help="Fills the form above with this run's problem, algorithm, configuration and budget.",
        key=f"solve_load_{run.run_id}",
    )
    if st.session_state.get("solve_load_error") == run.run_id:
        st.warning(
            "This run's settings cannot be restored: its algorithm or problem is not offered."
        )
    st.download_button(
        "Download request.yaml",
        (run.run_dir / "request.yaml").read_text(),
        file_name="request.yaml",
        mime="application/x-yaml",
        help="To run it outside Evolver-Studio: "
        f"java -cp Evolver-{EVOLVER_VERSION}-jar-with-dependencies.jar "
        "org.uma.evolver.cli.solving.SolveRunnerMain request.yaml, from the directory that holds "
        "resources/ (adjust outputDirectory).",
        key=f"solve_download_request_{run.run_id}",
    )
    st.download_button(
        "Download the fronts and indicators (zip)",
        zip_fronts(run.output_directory),
        file_name=f"{run.run_id}.zip",
        mime="application/zip",
        key=f"solve_download_{run.run_id}",
    )


def _render_history() -> None:
    """Choose a past run and show its results."""
    runs = {run.run_id: run for run in list_solve_runs(RUNS_DIRECTORY, WORKING_DIRECTORY)}
    if not runs:
        return
    st.subheader("Results")
    chosen = st.selectbox(
        "Previous runs",
        list(runs),
        index=None,
        placeholder="Choose a run to see its results",
        format_func=lambda run_id: (
            f"{run_id} · {runs[run_id].algorithm} · {runs[run_id].problem}"
            + (f" · {runs[run_id].state.value}" if runs[run_id].state else "")
        ),
        key="solve_history",
    )
    if chosen is not None:
        _render_results(runs[chosen])


st.title("Run algorithm")

jar = require_evolver_jar()
warn_if_jar_older_than_catalogue(jar)

run_in_progress = find_run_in_progress(RUNS_DIRECTORY)
if run_in_progress is not None:
    _render_run_in_progress(run_in_progress)
_render_form(can_run=run_in_progress is None)
_render_history()
