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
    require_evolver_jar,
    warn_if_jar_older_than_catalogue,
)
from evolver_studio.catalogue import BASE_ALGORITHMS, META_ALGORITHMS
from evolver_studio.evolver_client import (
    WORKING_DIRECTORY,
    RunState,
    RunStatus,
    cancel_training,
    read_pid,
    read_status,
    start_training,
    write_pid_file,
)
from evolver_studio.live_front import LiveFrontRenderer, build_front_figure
from evolver_studio.parameter_form import render_parameter_form
from evolver_studio.parameter_space import parse_parameter_space, serialize_parameter_space
from evolver_studio.request import (
    FLAT_META_SEARCH_SCALAR_KEYS,
    BaseLevelConfig,
    FlatMetaSearchConfig,
    base_level_to_yaml,
    flat_meta_search_to_yaml,
    parse_operator_flags_yaml,
    request_to_yaml,
)
from evolver_studio.resource_files import meta_optimizer_configuration_text, parameter_space_text
from evolver_studio.results import (
    deduplicate_consecutive_checkpoints,
    last_n_checkpoints,
    list_output_dir,
    load_indicators,
    read_metadata,
    read_results_pointer,
)
from evolver_studio.runs import RUNS_DIRECTORY_NAME, ActiveRun, find_active_run, mark_cancelled
from evolver_studio.slider_state import next_slider_value
from evolver_studio.training_set import (
    TrainingSet,
    default_training_set_table,
    parse_training_set,
)

DEFAULT_UPDATE_EVERY_EVALUATIONS = 100
LIVE_FRAGMENT_RUN_EVERY_SECONDS = 2


def _base_level_config(
    algorithm_name: str,
    encoding: str,
    yaml_parameter_space_file: str,
    extra_config: dict[str, str] | None,
    training_set: TrainingSet,
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
        training_problem_names=training_set.problem_names,
        training_reference_front_file_names=training_set.reference_front_file_names,
        training_evaluations=training_set.evaluations,
        indicator_names=["Epsilon", "NormalizedHypervolume"],
    )


def _meta_search_config(
    algorithm: str,
    meta_max_evaluations: int,
    number_of_cores: int,
    operator_flags: dict[str, object],
) -> FlatMetaSearchConfig:
    """Build the flat meta-search config for the selected meta-optimizer.

    The meta population size is left to Evolver's own default (50, see
    MetaAlgorithmRegistry.DEFAULT_POPULATION_SIZE), rather than duplicated here.

    Args:
        algorithm: The selected meta-optimizer's name (catalogue.MetaAlgorithm.name).
        meta_max_evaluations: Meta-level evaluation budget.
        number_of_cores: Cores used to parallelize base-level runs.
        operator_flags: The meta-optimizer's own operator configuration
            (possibly user-edited), as plain key/value pairs.

    Returns:
        The flat meta-search config for the selected algorithm.
    """
    return FlatMetaSearchConfig(
        algorithm=algorithm,
        meta_max_evaluations=meta_max_evaluations,
        # None omits metaPopulationSize, so Evolver applies its own default (50).
        meta_population_size=None,
        number_of_cores=number_of_cores,
        operator_flags=operator_flags,
    )


def _launch_run(
    jar: Path,
    run_id: str,
    algorithm_name: str,
    encoding: str,
    meta_algorithm_name: str,
    output_directory_base: str,
    parameter_space_text_: str,
    extra_config: dict[str, str] | None,
    training_set: TrainingSet,
    operator_flags: dict[str, object],
    meta_max_evaluations: int,
    number_of_cores: int,
    update_every_evaluations: int,
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
        operator_flags: The (possibly user-edited) meta-optimizer operator
            configuration, as plain key/value pairs.
        meta_max_evaluations: Meta-level evaluation budget.
        number_of_cores: Cores used to parallelize base-level runs.
        update_every_evaluations: Chosen live-preview redraw threshold, also
            used as writeFrequency/statusFrequency for this run.
    """
    run_dir = WORKING_DIRECTORY / RUNS_DIRECTORY_NAME / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    parameter_space_file = run_dir / "base_parameter_space.yaml"
    parameter_space_file.write_text(parameter_space_text_)

    base_level = _base_level_config(
        algorithm_name, encoding, str(parameter_space_file), extra_config, training_set
    )
    base_level_file = run_dir / "base_level.yaml"
    base_level_file.write_text(base_level_to_yaml(base_level))

    meta_search = _meta_search_config(
        meta_algorithm_name, meta_max_evaluations, number_of_cores, operator_flags
    )
    meta_search_file = run_dir / "meta_search.yaml"
    meta_search_file.write_text(flat_meta_search_to_yaml(meta_search))

    request_yaml = run_dir / "request.yaml"
    request_yaml.write_text(
        request_to_yaml(
            str(base_level_file),
            str(meta_search_file),
            f"{output_directory_base}/{run_id}",
            write_frequency=update_every_evaluations,
            status_frequency=update_every_evaluations,
        )
    )
    process = start_training(WORKING_DIRECTORY, jar, request_yaml, run_dir / "status.yaml")
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


def _sync_last_n_slider_value(slider_key: str, available: int) -> None:
    """Pre-seed the "last N" slider's session-state value before it's created.

    Called with no `value=` kwarg on the slider itself, since Streamlit
    re-applies `value=` whenever `max_value` changes (not only on first
    creation), which would silently override a value the user had chosen.

    Args:
        slider_key: The slider's widget key.
        available: The current number of distinct checkpoints to show by
            default, before the user narrows it.
    """
    tracked_key = f"{slider_key}_auto_tracked_available"
    next_value, next_tracked = next_slider_value(
        st.session_state.get(slider_key), st.session_state.get(tracked_key), available
    )
    st.session_state[slider_key] = next_value
    st.session_state[tracked_key] = next_tracked


def _render_front_with_slider(history: pd.DataFrame, slider_key: str, chart_key: str) -> None:
    """Draw the front evolution, letting the viewer narrow it to the last N checkpoints.

    Early checkpoints often have much larger indicator values than later,
    converged ones, which can swamp the late-stage detail in a combined plot.

    Args:
        history: Deduplicated checkpoint history to show.
        slider_key: Stable widget key so the chosen N persists across reruns.
        chart_key: Stable widget key for the chart itself.
    """
    available = history["Evaluation"].nunique()
    if available <= 1:
        # st.slider rejects min_value == max_value; nothing to narrow down yet anyway.
        st.plotly_chart(build_front_figure(history), width="stretch", key=chart_key)
        return
    _sync_last_n_slider_value(slider_key, available)
    n = st.slider("Show last N fronts", min_value=1, max_value=available, key=slider_key)
    figure = build_front_figure(last_n_checkpoints(history, n))
    st.plotly_chart(figure, width="stretch", key=chart_key)


def _poll_tick(
    status_yaml: Path,
    indicators_csv: Path,
    renderer: LiveFrontRenderer,
    interval: AdaptivePollInterval,
    slider_key: str,
) -> RunStatus | None:
    """Read the latest status, redraw the live front, and adapt the poll interval.

    Draws directly into whatever container is active when called — meant to
    run inside a fragment, which fully replaces its own contents each time it
    reruns, so both the progress bar and the front preview (from the last due
    history, not just a newly-due one, to avoid it flickering away between
    throttled redraws) are (re)drawn unconditionally on every call.

    Args:
        status_yaml: Path to the run's status file.
        indicators_csv: Path to the run's (still-growing) INDICATORS.csv.
        renderer: Tracks render-throttling state across polls.
        interval: Tracks the adaptive sleep interval across polls.
        slider_key: Stable widget key for the "last N checkpoints" slider.

    Returns:
        The latest parsed status, or None if not yet available.
    """
    status = read_status(status_yaml)
    if status is not None:
        fraction = status.evaluations_done / max(status.max_evaluations, 1)
        st.progress(fraction, text=f"{status.evaluations_done}/{status.max_evaluations}")
    else:
        st.progress(0.0, text="Starting…")
    update = renderer.poll(indicators_csv)
    interval.record_poll(update.changed, time.monotonic())
    if renderer.last_history is not None:
        _render_front_with_slider(renderer.last_history, slider_key, f"{slider_key}_chart")
    return status


def _render_indicator_front(indicators_csv: Path, run_id: str) -> None:
    """Plot the indicator front's evolution across all checkpoints written.

    Args:
        indicators_csv: Path to the completed run's INDICATORS.csv.
        run_id: The run's identifier, to key the "last N" slider.
    """
    history = load_indicators(indicators_csv)
    if history.empty:
        st.info("No indicator data was written.")
        return
    deduplicated = deduplicate_consecutive_checkpoints(history)
    _render_front_with_slider(
        deduplicated, f"final_last_n_slider_{run_id}", "final_indicator_front"
    )


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
        return
    pointer = read_results_pointer(run_dir / "results.yaml", WORKING_DIRECTORY)
    st.success("Training finished.")
    if st.checkbox("Show indicator front", value=True):
        _render_indicator_front(pointer.indicators_file, run_dir.name)
    _render_output_directory(pointer.output_directory, pointer.metadata_file)


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


def _default_operator_flags_text(jar: Path, example_config_file: str) -> str:
    """Read a meta-algorithm's example config, stripped to its operator flags.

    The example file under metaOptimizerConfigurations/ also carries algorithm/encoding/
    metaMaxEvaluations/metaPopulationSize/numberOfCores — those are set from other widgets in
    this app's launch form, not from this editor, so they're excluded here.

    Args:
        jar: Path to Evolver's jar.
        example_config_file: Filename under metaOptimizerConfigurations/ for the
            selected meta-algorithm (catalogue.MetaAlgorithm.example_config_file).

    Returns:
        A flat YAML mapping of just the operator flags, as starting text for the editor.
    """
    example = yaml.safe_load(meta_optimizer_configuration_text(jar, example_config_file))
    operator_flags = {
        key: value for key, value in example.items() if key not in FLAT_META_SEARCH_SCALAR_KEYS
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


def _render_training_set_editor(jar: Path) -> pd.DataFrame:
    """Let the user define the training set as an editable table of rows.

    Each row is one training problem, its reference front file, and its
    evaluation budget — BaseLevelConfig's three parallel lists, spelled out
    explicitly since the CLI does not resolve training sets by name.

    Args:
        jar: Path to Evolver's jar, to look up valid problem names from
            DescribeMain's manifest.

    Returns:
        The current (possibly user-edited) training set table.
    """
    problem_names = registered_problem_names(str(jar))
    problem_help = (
        "A curated name (e.g. ZDT4, DTLZ3) or a fully-qualified jMetal class name for any "
        "other Problem<S> on the classpath (e.g. "
        "org.uma.jmetal.problem.multiobjective.multiobjectivetsp.instance.KroAB100TSP), "
        "resolved by reflection."
    )
    if problem_names is not None:
        problem_help += " Curated names: " + ", ".join(sorted(problem_names))
    else:
        st.warning("Could not list registered problems (is Java installed?).")
    st.caption(
        "Reference front files live under resources/referenceFronts/ (or "
        "resources/referenceFrontsTSP/ for TSP problems); some "
        "problems use a dimension suffix (e.g. DTLZ1.3D.csv)."
    )
    return st.data_editor(
        st.session_state.get("training_set_table", default_training_set_table()),
        column_config={
            "problem": st.column_config.TextColumn("Problem", required=True, help=problem_help),
            "reference_front": st.column_config.TextColumn("Reference front file", required=True),
            "evaluations": st.column_config.NumberColumn(
                "Evaluations", min_value=1, step=100, required=True
            ),
        },
        num_rows="dynamic",
        width="stretch",
        key="training_set_table",
    )


def _render_active_run(active_run: ActiveRun) -> None:
    """Show progress and a live front preview for an in-progress run.

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

    @st.fragment(
        run_every=LIVE_FRAGMENT_RUN_EVERY_SECONDS, key=f"poll_fragment_{active_run.run_id}"
    )
    def _poll() -> None:
        status = _poll_tick(
            active_run.status_yaml,
            active_run.indicators_csv,
            renderer,
            interval,
            f"last_n_slider_{active_run.run_id}",
        )
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
    meta_max_evaluations = st.number_input("Meta max evaluations", value=2000, min_value=100)
    number_of_cores = st.number_input("Number of cores", value=8, min_value=1)
    update_every_evaluations = st.number_input(
        "Update every N evaluations",
        value=DEFAULT_UPDATE_EVERY_EVALUATIONS,
        min_value=1,
        step=100,
    )

    default_base_text = parameter_space_text(jar, algorithm.encodings[encoding])
    parameter_space_text_value = _render_parameter_space_editor(
        f"Base algorithm parameter space ({algorithm.name}, {encoding})",
        default_base_text,
        f"base_{algorithm.name}_{encoding}",
    )

    with st.expander(f"Meta-optimizer operator flags ({meta_algorithm.name})"):
        operator_flags = _render_operator_flags_editor(
            _default_operator_flags_text(jar, meta_algorithm.example_config_file),
            f"meta_operator_flags_{meta_algorithm.name}",
        )

    with st.expander("Training set (problems, reference fronts, evaluations)", expanded=True):
        training_set_table = _render_training_set_editor(jar)
    training_set = parse_training_set(training_set_table)
    if training_set is None:
        st.error("Every row needs a problem, a reference front file, and evaluations > 0.")

    can_launch = (
        parameter_space_text_value is not None
        and operator_flags is not None
        and training_set is not None
    )
    if can_launch:
        base_level = _base_level_config(
            algorithm.registry_name,
            encoding,
            "<written to disk at launch>",
            extra_config,
            training_set,
        )
        meta_search = _meta_search_config(
            meta_algorithm.name,
            int(meta_max_evaluations),
            int(number_of_cores),
            operator_flags or {},
        )
        st.code(base_level_to_yaml(base_level), language="yaml")
        st.code(flat_meta_search_to_yaml(meta_search), language="yaml")

    if st.button("Launch training", disabled=not can_launch):
        run_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        _launch_run(
            jar,
            run_id,
            algorithm.registry_name,
            encoding,
            meta_algorithm.name,
            output_directory_base,
            parameter_space_text_value,
            extra_config,
            training_set,
            operator_flags,
            int(meta_max_evaluations),
            int(number_of_cores),
            int(update_every_evaluations),
        )
        st.rerun()

    _render_last_finished_run_if_any()
