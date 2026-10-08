"""Training: configure, launch and monitor a training run.

Training runs as a detached subprocess, independent of this page's session —
matching the file-based request/status/results contract described in
CLAUDE.md, meant for long batch jobs that shouldn't need the UI to stay
connected. Reopening the app reconnects to an in-progress run instead of
losing track of it; a running job can be stopped with a Cancel button. The
polling loop uses `st.fragment` so it never blocks the rest of the page —
without it, a Cancel click couldn't be processed until the loop returned.
"""

import datetime as dt
import time
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

from evolver_studio.adaptive_poll import AdaptivePollInterval
from evolver_studio.app_state import (
    registered_problem_names,
    registered_problems,
    require_evolver_jar,
    warn_if_jar_older_than_catalogue,
)
from evolver_studio.catalogue import BASE_ALGORITHMS, META_ALGORITHMS, MetaAlgorithm
from evolver_studio.evolver_client import (
    WORKING_DIRECTORY,
    RunState,
    cancel_training,
    is_alive,
    read_pid,
    read_status,
    start_training,
    write_pid_file,
)
from evolver_studio.live_front import LiveFrontRenderer
from evolver_studio.monitor_view import (
    MonitorState,
    render_best_configurations,
    render_convergence,
    render_front_with_slider,
    render_indicator_front,
    render_log,
    render_overview,
    render_population,
    run_started_at,
    tab_names,
)
from evolver_studio.parameter_form import render_parameter_form
from evolver_studio.parameter_space import parse_parameter_space, serialize_parameter_space
from evolver_studio.problem_browser import render_problem_adder
from evolver_studio.problem_catalogue import problems_with_encoding
from evolver_studio.request import (
    DEFAULT_META_POPULATION_SIZE,
    META_SEARCH_SCALAR_KEYS,
    BaseLevelConfig,
    MetaSearchConfig,
    base_level_to_yaml,
    checkpoint_frequency_error,
    meta_search_to_yaml,
    parse_operator_flags_yaml,
    request_to_yaml,
)
from evolver_studio.resource_files import meta_optimizer_configuration_text, parameter_space_text
from evolver_studio.results import (
    list_output_dir,
    read_metadata,
    read_results_pointer,
)
from evolver_studio.runs import RUNS_DIRECTORY_NAME, ActiveRun, find_active_run, mark_cancelled
from evolver_studio.training_monitor import tail_text
from evolver_studio.training_set import (
    TRAINING_SET_COLUMNS,
    TrainingSet,
    parse_training_set,
    training_problem_specs,
    training_set_table,
)

DEFAULT_UPDATE_EVERY_EVALUATIONS = 100
PARAMETER_SPACE_FILE_NAME = "base_parameter_space.yaml"
LIVE_FRAGMENT_RUN_EVERY_SECONDS = 2


def _base_level_config(
    algorithm_name: str,
    encoding: str,
    yaml_parameter_space_file: str,
    extra_config: dict[str, str] | None,
    training_set: TrainingSet,
    problem_specs: list[str | dict],
) -> BaseLevelConfig:
    """Build the base-level config for the selected algorithm and training set.

    Indicators (Epsilon, NormalizedHypervolume) stay fixed for now; see
    ROADMAP.md for indicator selection as a later step.

    Args:
        algorithm_name: The exact string BaseAlgorithmRegistry.resolve() expects
            (catalogue.BaseAlgorithm.registry_name, e.g. "MOEAD", not "MOEA/D").
        encoding: The selected runnable encoding (catalogue.BaseAlgorithm.
            runnable_encodings), e.g. "Double" or "Permutation".
        yaml_parameter_space_file: Path to the base-level algorithm's parameter
            space YAML (possibly a run-specific, user-edited copy).
        extra_config: Algorithm-specific extra settings (e.g. MOEA/D's
            weightVectorFilesDirectory), or None when the algorithm needs none.
        training_set: The problems to train on, with their reference fronts
            and evaluation budgets (BaseLevelConfig's three parallel lists).
        problem_specs: The training set's problems as trainingProblemNames'
            entries (a name, or a `{class, args}` map), see
            training_set.training_problem_specs.

    Returns:
        The base-level config for the selected algorithm and training set.
    """
    return BaseLevelConfig(
        algorithm_name=algorithm_name,
        encoding=encoding,
        population_size=100,
        number_of_independent_runs=1,
        yaml_parameter_space_file=yaml_parameter_space_file,
        extra_config=extra_config,
        training_problem_names=problem_specs,
        training_reference_front_file_names=training_set.reference_front_file_names,
        training_evaluations=training_set.evaluations,
        indicator_names=["Epsilon", "NormalizedHypervolume"],
    )


def _meta_search_config(
    algorithm: str,
    meta_encoding: str,
    meta_max_evaluations: int,
    meta_max_minutes: float | None,
    meta_population_size: int | None,
    number_of_cores: int,
    operator_flags: dict[str, object],
) -> MetaSearchConfig:
    """Build the meta-search config for the selected meta-optimizer and encoding.

    Args:
        algorithm: The selected meta-optimizer's name (catalogue.MetaAlgorithm.name).
        meta_encoding: The meta-optimizer's encoding, "flat" or "tree".
        meta_max_evaluations: Meta-level evaluation budget.
        meta_max_minutes: Stop after this many minutes of computing time instead of after
            `meta_max_evaluations`, or None to stop by evaluations.
        meta_population_size: The meta-optimizer's population size, or None to leave it out of
            the file (Random Search has none).
        number_of_cores: Cores used to parallelize base-level runs.
        operator_flags: The meta-optimizer's own operator configuration
            (possibly user-edited), as plain key/value pairs.

    Returns:
        The meta-search config for the selected algorithm.
    """
    return MetaSearchConfig(
        algorithm=algorithm,
        meta_max_evaluations=meta_max_evaluations,
        meta_population_size=meta_population_size,
        number_of_cores=number_of_cores,
        operator_flags=operator_flags,
        encoding=meta_encoding,
        meta_max_computing_time_minutes=meta_max_minutes,
    )


def _launch_run(
    jar: Path,
    run_id: str,
    algorithm_name: str,
    encoding: str,
    meta_algorithm_name: str,
    meta_encoding: str,
    output_directory_base: str,
    parameter_space_text_: str,
    extra_config: dict[str, str] | None,
    training_set: TrainingSet,
    problem_specs: list[str | dict],
    operator_flags: dict[str, object],
    meta_max_evaluations: int,
    meta_max_minutes: float | None,
    meta_population_size: int | None,
    number_of_cores: int,
    update_every_evaluations: int,
    write_population: bool,
) -> None:
    """Write the base-level/meta-search/request files, launch training, persist its PID.

    A full request is three files (see evolver_studio/request.py): the
    base-level parameter space, the base-level and metaSearch configuration
    files, and request.yaml itself referencing the latter two by path.

    Args:
        jar: Path to Evolver's jar.
        run_id: This run's timestamp-based identifier.
        algorithm_name: The base algorithm's registry name (catalogue.py's
            BaseAlgorithm.registry_name), e.g. "MOEAD".
        encoding: The selected runnable encoding (catalogue.BaseAlgorithm.
            runnable_encodings), e.g. "Double" or "Permutation".
        meta_algorithm_name: The selected meta-optimizer's name
            (catalogue.MetaAlgorithm.name), e.g. "SPEA2".
        meta_encoding: The meta-optimizer's encoding, "flat" or "tree".
        output_directory_base: Output directory, nested under run_id so
            INDICATORS.csv (append-only in Evolver) never mixes checkpoints
            across runs that share the same base output directory.
        parameter_space_text_: The (possibly user-edited) base-level parameter
            space YAML, written to this run's own file — Evolver accepts an
            absolute path here regardless of its working directory.
        extra_config: Algorithm-specific extra settings (e.g. MOEA/D's
            weightVectorFilesDirectory), or None.
        training_set: The problems to train on, with their reference fronts
            and evaluation budgets.
        problem_specs: The training set's problems as trainingProblemNames'
            entries.
        operator_flags: The (possibly user-edited) meta-optimizer operator
            configuration, as plain key/value pairs.
        meta_max_evaluations: Meta-level evaluation budget.
        meta_max_minutes: The computing time limit in minutes, when the meta-optimizer stops by
            time instead of by evaluations; else None.
        meta_population_size: The meta-optimizer's population size, or None when it has none.
        number_of_cores: Cores used to parallelize base-level runs.
        write_population: Whether Evolver also writes the meta-optimizer's whole population at
            every checkpoint, for the population viewer of the monitor.
        update_every_evaluations: Chosen live-preview redraw threshold, also
            used as writeFrequency/statusFrequency for this run.
    """
    run_dir = WORKING_DIRECTORY / RUNS_DIRECTORY_NAME / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    parameter_space_file = run_dir / PARAMETER_SPACE_FILE_NAME
    parameter_space_file.write_text(parameter_space_text_)

    base_level = _base_level_config(
        algorithm_name,
        encoding,
        str(parameter_space_file),
        extra_config,
        training_set,
        problem_specs,
    )
    base_level_file = run_dir / "base_level.yaml"
    base_level_file.write_text(base_level_to_yaml(base_level))

    meta_search = _meta_search_config(
        meta_algorithm_name,
        meta_encoding,
        meta_max_evaluations,
        meta_max_minutes,
        meta_population_size,
        number_of_cores,
        operator_flags,
    )
    meta_search_file = run_dir / "meta_search.yaml"
    meta_search_file.write_text(meta_search_to_yaml(meta_search))

    request_yaml = run_dir / "request.yaml"
    request_yaml.write_text(
        request_to_yaml(
            str(base_level_file),
            str(meta_search_file),
            f"{output_directory_base}/{run_id}",
            write_frequency=update_every_evaluations,
            status_frequency=update_every_evaluations,
            write_population=write_population,
        )
    )
    process = start_training(
        WORKING_DIRECTORY, jar, request_yaml, run_dir / "status.yaml", run_dir / "runner.log"
    )
    write_pid_file(run_dir / "pid.txt", process.pid)
    st.session_state[f"update_every_evaluations_{run_id}"] = update_every_evaluations
    st.session_state.pop("last_finished_run", None)


def _cancel(active_run: ActiveRun) -> None:
    """Terminate a run's subprocess and mark it as no longer active.

    Args:
        active_run: The run to cancel.
    """
    pid = read_pid(active_run.pid_file)
    if pid is not None:
        cancel_training(pid)
    mark_cancelled(active_run.run_dir)


def _live_state(
    run_id: str, update_every_evaluations: int
) -> tuple[LiveFrontRenderer, AdaptivePollInterval]:
    """Get or create a run's live-preview state, persisted across fragment reruns.

    Args:
        run_id: The run's identifier, used to key the stashed state.
        update_every_evaluations: Redraw threshold used only when first created.

    Returns:
        The renderer and poll interval tracking this run's live preview.
    """
    key = f"live_state_{run_id}"
    if key not in st.session_state:
        st.session_state[key] = (
            LiveFrontRenderer(update_every_evaluations),
            AdaptivePollInterval(),
        )
    return st.session_state[key]


def _render_front_tab(
    indicators_csv: Path,
    renderer: LiveFrontRenderer,
    interval: AdaptivePollInterval,
    slider_key: str,
) -> None:
    """Redraw the live front and adapt the poll interval.

    Draws directly into whatever container is active when called — meant to run inside a
    fragment, which fully replaces its own contents each time it reruns, so the front preview is
    redrawn from the last due history, not just a newly-due one, to avoid it flickering away
    between throttled redraws.

    Args:
        indicators_csv: Path to the run's (still-growing) INDICATORS.csv.
        renderer: Tracks the checkpoint history available for display, throttled.
        interval: Tracks how long to sleep before the next poll.
        slider_key: Stable widget key for the "last N checkpoints" slider.
    """
    update = renderer.poll(indicators_csv)
    interval.record_poll(update.changed, time.monotonic())
    if renderer.last_history is not None:
        render_front_with_slider(renderer.last_history, slider_key, f"{slider_key}_chart")
    else:
        st.caption("No checkpoint yet: Evolver writes the first one after the update frequency.")


def _render_output_directory(output_directory: Path, metadata_file: Path) -> None:
    """Show the final results folder: absolute path, file listing, METADATA.txt.

    Args:
        output_directory: The run's output directory.
        metadata_file: Path to METADATA.txt inside it.
    """
    st.write(f"Results folder: `{output_directory}`")
    for name, size in list_output_dir(output_directory):
        st.write(f"- {name} ({size} bytes)")
    with st.expander("METADATA.txt"):
        st.text(read_metadata(metadata_file))


def _render_last_finished_run_if_any() -> None:
    """Show the most recently completed run's results, if one is pending display."""
    pending = st.session_state.get("last_finished_run")
    if pending is None:
        return
    status, run_dir = pending
    if status.state == RunState.FAILED:
        st.error(status.error_message)
        _render_finished_log(run_dir)
        return
    pointer = read_results_pointer(run_dir / "results.yaml", WORKING_DIRECTORY)
    st.success("Training finished.")
    state = MonitorState(
        run_started_at(run_dir.name),
        pointer.indicators_file,
        pointer.population_indicators_file,
        pointer.var_conf_file,
        run_dir / "runner.log",
    )
    state.poll()
    key = f"finished_{run_dir.name}"
    names = ["Front", "Convergence"]
    if state.population is not None:
        names.append("Population")
    names += ["Best configurations", "Files"]
    tabs = iter(st.tabs(names))
    with next(tabs):
        render_indicator_front(pointer.indicators_file, run_dir.name)
    with next(tabs):
        render_convergence(state, f"{key}_convergence")
    if state.population is not None:
        with next(tabs):
            render_population(state, f"{key}_population")
    with next(tabs):
        render_best_configurations(state, f"{key}_best")
    with next(tabs):
        _render_output_directory(pointer.output_directory, pointer.metadata_file)
        render_log(state)


def _render_finished_log(run_dir: Path) -> None:
    """Show the log of a run that failed, which says why."""
    text = tail_text(run_dir / "runner.log", 40)
    if text:
        with st.expander("Runner output", expanded=True):
            st.code(text, language=None, wrap_lines=True)


def _render_expert_editor(default_text: str, key: str) -> str | None:
    """A raw YAML text area, validated by re-parsing on every change.

    Args:
        default_text: Text to pre-fill the text area with, the first time
            it's shown.
        key: Stable session-state key for this text area.

    Returns:
        The current text if it parses as a valid parameter space, else None
        (a validation error is already shown).
    """
    if key not in st.session_state:
        st.session_state[key] = default_text
    text = st.text_area("Parameter space YAML", key=key, height=300)
    try:
        parse_parameter_space(text)
    except (ValueError, KeyError, yaml.YAMLError) as error:
        st.error(f"Invalid YAML: {error}")
        return None
    return text


def _render_guided_editor(default_text: str, key_prefix: str) -> str:
    """A dynamic form built from the default parameter space.

    Args:
        default_text: The parameter space YAML to build the form from.
        key_prefix: Prefix for the form's widget keys.

    Returns:
        The edited parameter space, serialized back to YAML.
    """
    parameters = parse_parameter_space(default_text)
    edited = render_parameter_form(parameters, key_prefix)
    return serialize_parameter_space(edited)


META_ENCODINGS = {"Flat": "flat", "Tree": "tree"}
STOP_BY_EVALUATIONS = "Evaluations"
STOP_BY_TIME = "Computing time"
DEFAULT_META_EVALUATIONS = 2000
DEFAULT_META_MINUTES = 10.0


def _default_update_every(population_size: int) -> int:
    """The multiple of the meta population size closest to the default update frequency."""
    return population_size * max(round(DEFAULT_UPDATE_EVERY_EVALUATIONS / population_size), 1)


def _render_meta_population(meta_algorithm: MetaAlgorithm) -> int | None:
    """Choose the size of the meta-optimizer's population.

    Args:
        meta_algorithm: The selected meta-optimizer.

    Returns:
        The size, or None for a meta-optimizer that has no population (Random Search).
    """
    if not meta_algorithm.uses_population:
        st.caption(f"{meta_algorithm.name} has no population: it samples configurations at random.")
        return None
    return int(
        st.number_input(
            "Meta-optimizer population size",
            value=DEFAULT_META_POPULATION_SIZE,
            min_value=2,
            step=10,
            key=f"train_meta_population_{meta_algorithm.name}",
            help="The configurations the meta-optimizer evolves at once; its offspring is as "
            "large. A larger population explores more but needs more evaluations to converge. "
            "Evolver's default is 50.",
        )
    )


def _render_meta_limit() -> tuple[int, float | None]:
    """Choose when the meta-optimizer stops: after a number of evaluations or a computing time.

    Evolver takes one limit or the other, never both.

    Returns:
        The meta-evaluations to run (the default one when the limit is by time) and the minutes
        of computing time (None when the limit is by evaluations).
    """
    stop_by = st.radio(
        "Stop the meta-optimizer after",
        [STOP_BY_EVALUATIONS, STOP_BY_TIME],
        horizontal=True,
        key="train_meta_stop_by",
        help="A number of meta-evaluations (configurations evaluated), or an amount of computing "
        "time. With a time limit the generation in progress is completed, so the run lasts a "
        "little longer than the limit; and what a run does in that time depends on the "
        "meta-optimizer and on the number of cores.",
    )
    if stop_by == STOP_BY_TIME:
        minutes = st.number_input(
            "Meta max computing time (minutes)",
            value=DEFAULT_META_MINUTES,
            min_value=0.1,
            step=1.0,
            key="train_meta_minutes",
        )
        return DEFAULT_META_EVALUATIONS, float(minutes)
    evaluations = st.number_input(
        "Meta max evaluations",
        value=DEFAULT_META_EVALUATIONS,
        min_value=100,
        key="train_meta_evals",
    )
    return int(evaluations), None


def _render_meta_encoding(meta_algorithm: MetaAlgorithm) -> str:
    """Choose the meta-optimizer's encoding: flat, or tree when the algorithm supports it.

    The flat encoding searches a vector of numbers in [0, 1]; the tree encoding searches
    derivation trees of the base algorithm's grammar. Each has its own operators, so the flags
    below depend on the choice.

    Args:
        meta_algorithm: The selected meta-optimizer.

    Returns:
        "flat" or "tree".
    """
    supports_tree = meta_algorithm.supports_tree and meta_algorithm.tree_example_config_file
    choice = st.radio(
        "Meta-optimizer encoding",
        list(META_ENCODINGS) if supports_tree else ["Flat"],
        horizontal=True,
        key=f"train_meta_encoding_{meta_algorithm.name}",
        help="Flat: a vector of numbers in [0, 1], one for each parameter of the base algorithm. "
        "Tree: a derivation tree of the grammar of its parameter space, which has no inactive "
        "parameters. Each has its own operators.",
    )
    if not supports_tree:
        reason = meta_algorithm.flat_only_reason
        st.caption(
            f"{meta_algorithm.name} supports only the flat encoding"
            + (f": {reason}." if reason else ".")
        )
    return META_ENCODINGS[choice]


def _default_operator_flags_text(jar: Path, example_config_file: str) -> str:
    """Read a meta-algorithm's example config, stripped to its operator flags.

    The example file under metaOptimizerConfigurations/ also carries algorithm/encoding/
    metaMaxEvaluations/metaPopulationSize/numberOfCores — those are set from other widgets in
    this app's launch form, not from this editor, so they're excluded here.

    Args:
        jar: Path to Evolver's jar.
        example_config_file: Filename under metaOptimizerConfigurations/ for the
            selected meta-algorithm and encoding (catalogue.MetaAlgorithm.example_config_file or
            tree_example_config_file).

    Returns:
        A flat YAML mapping of just the operator flags, as starting text for the editor.
    """
    example = yaml.safe_load(meta_optimizer_configuration_text(jar, example_config_file))
    operator_flags = {
        key: value for key, value in example.items() if key not in META_SEARCH_SCALAR_KEYS
    }
    return yaml.safe_dump(operator_flags, sort_keys=False)


def _render_operator_flags_editor(default_text: str, key: str) -> dict[str, object] | None:
    """A raw YAML text area for the meta-optimizer's operator flags, validated on every change.

    Args:
        default_text: Text to pre-fill the text area with, the first time
            it's shown.
        key: Stable session-state key for this text area.

    Returns:
        The parsed operator flags if the text is valid, else None (a
        validation error is already shown).
    """
    if key not in st.session_state:
        st.session_state[key] = default_text
    text = st.text_area("Operator flags (YAML)", key=key, height=200)
    try:
        return parse_operator_flags_yaml(text)
    except (ValueError, yaml.YAMLError) as error:
        st.error(f"Invalid YAML: {error}")
        return None


def _render_parameter_space_editor(title: str, default_text: str, key_prefix: str) -> str | None:
    """Let the user pick or edit a parameter space, guided or expert.

    Args:
        title: This editor's expander title.
        default_text: The default parameter space YAML's text.
        key_prefix: Prefix for this editor's widget keys — unique per context
            (base-level vs. meta-level, and per selected algorithm) so
            switching algorithms doesn't carry over stale edited text or
            form state from a different parameter space.

    Returns:
        The chosen parameter space's YAML text, or None if invalid (expert
        mode only — the guided form can't produce invalid YAML).
    """
    with st.expander(title):
        mode = st.radio("Mode", ["Guided", "Expert"], horizontal=True, key=f"{key_prefix}_mode")
        if mode == "Expert":
            return _render_expert_editor(default_text, f"{key_prefix}_expert_text")
        return _render_guided_editor(default_text, f"{key_prefix}_form")


DEFAULT_TRAINING_PROBLEMS = {"Double": ["ZDT4"]}
TRAINING_SET_ROWS_KEY = "training_set_rows"


def _render_training_set_editor(jar: Path, encoding: str) -> pd.DataFrame:
    """Let the user choose the training problems, and edit what each one needs.

    The problems are chosen from a list (those of the base algorithm's encoding when the jar
    describes them), and a fully-qualified class name can be typed to add any other one; each
    chosen problem is a row of a table with its arguments, its reference front file and its
    evaluation budget — BaseLevelConfig's parallel lists, spelled out explicitly since the CLI
    does not resolve training sets by name.

    Args:
        jar: Path to Evolver's jar, to look up valid problem names from
            DescribeMain's manifest.
        encoding: The base algorithm's encoding: the problems offered are those of it, when the
            jar describes them.

    Returns:
        The training set table (without rows while no problem is chosen).
    """
    catalogue = registered_problems(str(jar))
    options = (
        problems_with_encoding(catalogue, encoding)
        if catalogue is not None
        else registered_problem_names(str(jar))
    )
    if options is None:
        st.warning("Could not list registered problems (is Java installed?).")
        options = []
    problems_key = f"train_problems_{encoding}"
    names = st.multiselect(
        "Training problems",
        options,
        default=[n for n in DEFAULT_TRAINING_PROBLEMS.get(encoding, []) if n in options],
        accept_new_options=True,
        placeholder="Choose the problems, or type a class name",
        key=problems_key,
        help="The problems of this encoding. To train on any other jMetal Problem on the "
        "classpath, type its fully-qualified class name (e.g. org.uma.jmetal.problem."
        "multiobjective.multiobjectivetsp.instance.KroAB100TSP): Evolver resolves it by "
        "reflection.",
    )
    if catalogue is not None:
        render_problem_adder(catalogue, encoding, problems_key, "train")
    if not names:
        return pd.DataFrame(columns=list(TRAINING_SET_COLUMNS))
    st.caption(
        "Reference front files live under resources/referenceFronts/ (or "
        "resources/referenceFrontsTSP/ for TSP problems); some "
        "problems use a dimension suffix (e.g. DTLZ1.3D.csv)."
    )
    table = training_set_table(
        names,
        catalogue or {},
        WORKING_DIRECTORY,
        st.session_state.get(TRAINING_SET_ROWS_KEY),
    )
    edited = st.data_editor(
        table,
        column_config={
            "problem": st.column_config.TextColumn("Problem", disabled=True),
            "arguments": st.column_config.TextColumn(
                "Arguments",
                help="Optional: the problem's constructor arguments, all of them or none, "
                "comma-separated (e.g. 12, 2 for DTLZ2 with 12 variables and 2 objectives). "
                "Explore › Problems lists each problem's arguments.",
            ),
            "reference_front": st.column_config.TextColumn("Reference front file", required=True),
            "evaluations": st.column_config.NumberColumn(
                "Evaluations", min_value=1, step=100, required=True
            ),
        },
        hide_index=True,
        width="stretch",
        key=f"train_set_table_{encoding}_{'|'.join(names)}",
    )
    st.session_state[TRAINING_SET_ROWS_KEY] = edited[list(TRAINING_SET_COLUMNS)]
    return edited


def _monitor_state(active_run: ActiveRun) -> MonitorState:
    """Get or create what the monitor of a run remembers, persisted across fragment reruns."""
    key = f"monitor_{active_run.run_id}"
    if key not in st.session_state:
        st.session_state[key] = MonitorState(
            run_started_at(active_run.run_id),
            active_run.indicators_csv,
            active_run.population_indicators_csv,
            active_run.indicators_csv.parent / "VAR_CONF.txt",
            active_run.log_file,
        )
    return st.session_state[key]


def _render_active_run(active_run: ActiveRun) -> None:
    """Monitor an in-progress run: its progress, front, population, convergence and log.

    Args:
        active_run: The run currently in progress.
    """
    st.info(f"Training in progress (run {active_run.run_id}).")
    if st.button("Cancel training"):
        _cancel(active_run)
        st.rerun()

    default_n = st.session_state.get(
        f"update_every_evaluations_{active_run.run_id}", DEFAULT_UPDATE_EVERY_EVALUATIONS
    )
    renderer, interval = _live_state(active_run.run_id, default_n)
    monitor = _monitor_state(active_run)
    key = f"monitor_{active_run.run_id}"

    @st.fragment(
        run_every=LIVE_FRAGMENT_RUN_EVERY_SECONDS, key=f"poll_fragment_{active_run.run_id}"
    )
    def _poll() -> None:
        status = read_status(active_run.status_yaml)
        if status is not None:
            monitor.record(status)
        monitor.poll()
        pid = read_pid(active_run.pid_file)
        tabs = st.tabs(tab_names(monitor.population is not None))
        names = iter(tabs)
        with next(names):
            render_overview(monitor, status, pid is not None and is_alive(pid))
        with next(names):
            _render_front_tab(
                active_run.indicators_csv, renderer, interval, f"last_n_slider_{active_run.run_id}"
            )
        if monitor.population is not None:
            with next(names):
                render_population(monitor, f"{key}_population")
        with next(names):
            render_convergence(monitor, f"{key}_convergence")
        with next(names):
            render_best_configurations(monitor, f"{key}_best")
        with next(names):
            render_log(monitor)
        if status is not None and status.state != RunState.RUNNING:
            st.session_state["last_finished_run"] = (status, active_run.run_dir)
            st.rerun()

    _poll()


st.title("Training")

jar = require_evolver_jar()
warn_if_jar_older_than_catalogue(jar)

active_run = find_active_run(WORKING_DIRECTORY)

if active_run is not None:
    _render_active_run(active_run)
else:
    runnable_algorithms = [a for a in BASE_ALGORITHMS if a.runnable_today]
    selected_algorithm_name = st.selectbox(
        "Base algorithm", [a.name for a in runnable_algorithms], key="train_base_algorithm"
    )
    algorithm = next(a for a in runnable_algorithms if a.name == selected_algorithm_name)
    encoding = st.selectbox(
        "Encoding", algorithm.runnable_encodings, key=f"train_encoding_{algorithm.name}"
    )

    extra_config = None
    if "weightVectorFilesDirectory" in algorithm.required_extra_config_keys:
        weight_vectors_directory = st.text_input(
            "Weight vector files directory", "resources/weightVectors"
        )
        extra_config = {"weightVectorFilesDirectory": weight_vectors_directory}

    wired_meta_algorithms = [m for m in META_ALGORITHMS if m.wired_into_cli_runner]
    selected_meta_algorithm_name = st.selectbox(
        "Meta-optimizer", [m.name for m in wired_meta_algorithms], key="train_meta_algorithm"
    )
    meta_algorithm = next(
        m for m in wired_meta_algorithms if m.name == selected_meta_algorithm_name
    )

    output_directory_base = st.text_input("Output directory", "results/nsgaii/ZDT4")
    meta_max_evaluations, meta_max_minutes = _render_meta_limit()
    meta_population_size = _render_meta_population(meta_algorithm)
    number_of_cores = st.number_input("Number of cores", value=8, min_value=1)
    checkpoint_unit = meta_population_size or DEFAULT_META_POPULATION_SIZE
    update_every_evaluations = st.number_input(
        "Update every N evaluations",
        value=_default_update_every(checkpoint_unit),
        min_value=1,
        step=checkpoint_unit,
        key=f"train_update_every_{checkpoint_unit}",
        help="How often Evolver writes its result files and its status, and so how often the "
        "monitor below can show something new. It must be a multiple of the meta-optimizer's "
        f"population size ({checkpoint_unit}).",
    )
    frequency_error = checkpoint_frequency_error(int(update_every_evaluations), checkpoint_unit)
    if frequency_error is not None:
        st.error(frequency_error)
    write_population = st.checkbox(
        "Show the meta-optimizer's population while it runs",
        key="train_write_population",
        help="Evolver also writes the whole population of the meta-optimizer at every "
        "checkpoint, not only its non-dominated front, and the monitor gets a Population tab "
        "that shows how it evolves. The files are larger; leave it off if you do not need it.",
    )

    default_base_text = parameter_space_text(jar, algorithm.encodings[encoding])
    parameter_space_text_value = _render_parameter_space_editor(
        f"Base algorithm parameter space ({algorithm.name}, {encoding})",
        default_base_text,
        f"base_{algorithm.name}_{encoding}",
    )

    meta_encoding = _render_meta_encoding(meta_algorithm)
    example_file = (
        meta_algorithm.tree_example_config_file
        if meta_encoding == "tree"
        else meta_algorithm.example_config_file
    )
    with st.expander(f"Meta-optimizer operator flags ({meta_algorithm.name}, {meta_encoding})"):
        if meta_algorithm.fixed_operators_note:
            st.info(meta_algorithm.fixed_operators_note)
        operator_flags = _render_operator_flags_editor(
            _default_operator_flags_text(jar, example_file),
            f"meta_operator_flags_{meta_algorithm.name}_{meta_encoding}",
        )

    with st.expander("Training set (problems, reference fronts, evaluations)", expanded=True):
        training_set_table_value = _render_training_set_editor(jar, encoding)
    training_set = parse_training_set(training_set_table_value)
    problem_specs: list[str | dict] = []
    problem_errors: list[str] = []
    if training_set_table_value.empty:
        st.error("Choose at least one training problem.")
    elif training_set is None:
        st.error("Every row needs a reference front file and evaluations > 0.")
    else:
        problem_specs, problem_errors = training_problem_specs(
            training_set, registered_problems(str(jar)), encoding
        )
        for error in problem_errors:
            st.error(error)

    can_launch = (
        frequency_error is None
        and parameter_space_text_value is not None
        and operator_flags is not None
        and training_set is not None
        and not problem_errors
    )
    if can_launch:
        base_level = _base_level_config(
            algorithm.registry_name,
            encoding,
            f"<run folder>/{PARAMETER_SPACE_FILE_NAME}",
            extra_config,
            training_set,
            problem_specs,
        )
        meta_search = _meta_search_config(
            meta_algorithm.name,
            meta_encoding,
            meta_max_evaluations,
            meta_max_minutes,
            meta_population_size,
            int(number_of_cores),
            operator_flags or {},
        )
        st.caption(
            "The two files Evolver will read, as written into the run's folder when you launch. "
            f"`yamlParameterSpaceFile` is the base algorithm's parameter space as edited above, "
            f"saved there as `{PARAMETER_SPACE_FILE_NAME}`: the folder is named after the launch "
            "time, so its path is only known then."
        )
        st.code(base_level_to_yaml(base_level), language="yaml")
        st.code(meta_search_to_yaml(meta_search), language="yaml")

    if st.button("Launch training", disabled=not can_launch):
        run_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        _launch_run(
            jar,
            run_id,
            algorithm.registry_name,
            encoding,
            meta_algorithm.name,
            meta_encoding,
            output_directory_base,
            parameter_space_text_value,
            extra_config,
            training_set,
            problem_specs,
            operator_flags,
            meta_max_evaluations,
            meta_max_minutes,
            meta_population_size,
            int(number_of_cores),
            int(update_every_evaluations),
            write_population,
        )
        st.rerun()

    _render_last_finished_run_if_any()
