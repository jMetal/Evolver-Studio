"""Entrenamiento: configure, launch and monitor a training run.

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
from evolver_studio.app_state import render_sidebar
from evolver_studio.catalogue import BASE_ALGORITHMS
from evolver_studio.evolver_client import (
    RunState,
    RunStatus,
    cancel_training,
    jar_path,
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
from evolver_studio.runs import ActiveRun, find_active_run, mark_cancelled
from evolver_studio.slider_state import next_slider_value

DEFAULT_UPDATE_EVERY_EVALUATIONS = 100
LIVE_FRAGMENT_RUN_EVERY_SECONDS = 2
# The single meta-optimizer exposed in this app's launch form for now. MetaAlgorithmRegistry
# also registers SPEA2/SMPSO/AsyncNSGA-II for the flat encoding (see catalogue.py), but each
# needs its own operator-flags editing UI to be genuinely useful — deferred, see ROADMAP.md.
LAUNCHABLE_META_ALGORITHM_NAME = "ParallelNSGA-II"
LAUNCHABLE_META_ALGORITHM_EXAMPLE_CONFIG_FILE = "MetaParallelNSGAIIFlatConfiguration.yaml"


def _base_level_config(
    algorithm_name: str,
    yaml_parameter_space_file: str,
    extra_config: dict[str, str] | None,
) -> BaseLevelConfig:
    """Build the base-level config for the selected algorithm, tuned on ZDT4.

    The training problem (ZDT4) and indicators (Epsilon, NormalizedHypervolume)
    stay fixed for now — only the algorithm and its parameter space are chosen
    by the user; see ROADMAP.md for problem/indicator selection as a later step.

    Args:
        algorithm_name: The exact string BaseAlgorithmRegistry.resolve() expects
            (catalogue.BaseAlgorithm.registry_name, e.g. "MOEAD", not "MOEA/D").
        yaml_parameter_space_file: Path to the base-level algorithm's parameter
            space YAML (possibly a run-specific, user-edited copy).
        extra_config: Algorithm-specific extra settings (e.g. MOEA/D's
            weightVectorFilesDirectory), or None when the algorithm needs none.

    Returns:
        The base-level config for the selected algorithm tuned on ZDT4.
    """
    return BaseLevelConfig(
        algorithm_name=algorithm_name,
        population_size=100,
        number_of_independent_runs=1,
        yaml_parameter_space_file=yaml_parameter_space_file,
        extra_config=extra_config,
        training_problem_names=["ZDT4"],
        training_reference_front_file_names=["resources/referenceFronts/ZDT4.csv"],
        training_evaluations=[12000],
        indicator_names=["Epsilon", "NormalizedHypervolume"],
    )


def _meta_search_config(
    meta_max_evaluations: int, number_of_cores: int, operator_flags: dict[str, object]
) -> FlatMetaSearchConfig:
    """Build the flat meta-search config for the launchable meta-optimizer.

    Args:
        meta_max_evaluations: Meta-level evaluation budget.
        number_of_cores: Cores used to parallelize base-level runs.
        operator_flags: The meta-optimizer's own operator configuration
            (possibly user-edited), as plain key/value pairs.

    Returns:
        The flat meta-search config for LAUNCHABLE_META_ALGORITHM_NAME.
    """
    return FlatMetaSearchConfig(
        algorithm=LAUNCHABLE_META_ALGORITHM_NAME,
        meta_max_evaluations=meta_max_evaluations,
        meta_population_size=100,
        number_of_cores=number_of_cores,
        operator_flags=operator_flags,
    )


def _launch_run(
    evolver_home: Path,
    run_id: str,
    algorithm_name: str,
    output_directory_base: str,
    parameter_space_text_: str,
    extra_config: dict[str, str] | None,
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
        evolver_home: Path to the Evolver checkout (JVM working directory).
        run_id: This run's timestamp-based identifier.
        algorithm_name: The base algorithm's registry name (catalogue.py's
            BaseAlgorithm.registry_name), e.g. "MOEAD".
        output_directory_base: Output directory, nested under run_id so
            INDICATORS.csv (append-only in Evolver) never mixes checkpoints
            across runs that share the same base output directory.
        parameter_space_text_: The (possibly user-edited) base-level parameter
            space YAML, written to this run's own file — Evolver accepts an
            absolute path here regardless of its working directory.
        extra_config: Algorithm-specific extra settings (e.g. MOEA/D's
            weightVectorFilesDirectory), or None.
        operator_flags: The (possibly user-edited) meta-optimizer operator
            configuration, as plain key/value pairs.
        meta_max_evaluations: Meta-level evaluation budget.
        number_of_cores: Cores used to parallelize base-level runs.
        update_every_evaluations: Chosen live-preview redraw threshold, also
            used as writeFrequency/statusFrequency for this run.
    """
    run_dir = evolver_home / "cli-runner-runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    parameter_space_file = run_dir / "base_parameter_space.yaml"
    parameter_space_file.write_text(parameter_space_text_)

    base_level = _base_level_config(algorithm_name, str(parameter_space_file), extra_config)
    base_level_file = run_dir / "base_level.yaml"
    base_level_file.write_text(base_level_to_yaml(base_level))

    meta_search = _meta_search_config(meta_max_evaluations, number_of_cores, operator_flags)
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
    process = start_training(
        evolver_home, jar_path(evolver_home), request_yaml, run_dir / "status.yaml"
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
    n = st.slider("Mostrar últimos N frentes", min_value=1, max_value=available, key=slider_key)
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


def _render_last_finished_run_if_any(evolver_home: Path) -> None:
    """Show the most recently completed run's results, if one is pending display.

    Args:
        evolver_home: Path to the Evolver checkout, to resolve results.yaml's paths.
    """
    pending = st.session_state.get("last_finished_run")
    if pending is None:
        return
    status, run_dir = pending
    if status.state == RunState.FAILED:
        st.error(status.error_message)
        return
    pointer = read_results_pointer(run_dir / "results.yaml", evolver_home)
    st.success("Training finished.")
    if st.checkbox("Mostrar frente de indicadores", value=True):
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
    text = st.text_area("YAML del espacio de parámetros", key=key, height=300)
    try:
        parse_parameter_space(text)
    except (ValueError, KeyError, yaml.YAMLError) as error:
        st.error(f"YAML inválido: {error}")
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


def _default_operator_flags_text(evolver_home: Path) -> str:
    """Read the launchable meta-algorithm's example config, stripped to its operator flags.

    The example file under metaOptimizerConfigurations/ also carries algorithm/encoding/
    metaMaxEvaluations/metaPopulationSize/numberOfCores — those are set from other widgets in
    this app's launch form, not from this editor, so they're excluded here.

    Args:
        evolver_home: Path to the Evolver checkout.

    Returns:
        A flat YAML mapping of just the operator flags, as starting text for the editor.
    """
    example = yaml.safe_load(
        meta_optimizer_configuration_text(
            evolver_home, LAUNCHABLE_META_ALGORITHM_EXAMPLE_CONFIG_FILE
        )
    )
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
        st.error(f"YAML inválido: {error}")
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
        mode = st.radio("Modo", ["Guiado", "Experto"], horizontal=True, key=f"{key_prefix}_mode")
        if mode == "Experto":
            return _render_expert_editor(default_text, f"{key_prefix}_expert_text")
        return _render_guided_editor(default_text, f"{key_prefix}_form")


def _render_active_run(evolver_home: Path, active_run: ActiveRun) -> None:
    """Show progress and a live front preview for an in-progress run.

    Args:
        evolver_home: Path to the Evolver checkout, to resolve results.yaml's paths.
        active_run: The run currently in progress.
    """
    st.info(f"Entrenamiento en curso (run {active_run.run_id}).")
    if st.button("Cancelar entrenamiento"):
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


st.title("Entrenamiento")

evolver_home = render_sidebar()

active_run = find_active_run(evolver_home)

if active_run is not None:
    _render_active_run(evolver_home, active_run)
else:
    runnable_algorithms = [a for a in BASE_ALGORITHMS if a.runnable_today]
    selected_algorithm_name = st.selectbox(
        "Algoritmo base", [a.name for a in runnable_algorithms], key="train_base_algorithm"
    )
    algorithm = next(a for a in runnable_algorithms if a.name == selected_algorithm_name)
    st.caption(
        f"Codificación: {algorithm.runnable_encoding} — la única que "
        "BaseAlgorithmRegistry construye hoy, sea cual sea el YAML elegido."
    )

    extra_config = None
    if algorithm.name == "MOEA/D":
        weight_vectors_directory = st.text_input(
            "Weight vector files directory", "resources/weightVectors"
        )
        extra_config = {"weightVectorFilesDirectory": weight_vectors_directory}

    st.selectbox("Meta-optimizador", [LAUNCHABLE_META_ALGORITHM_NAME], key="train_meta_algorithm")

    output_directory_base = st.text_input("Output directory", "results/nsgaii/ZDT4")
    meta_max_evaluations = st.number_input("Meta max evaluations", value=2000, min_value=100)
    number_of_cores = st.number_input("Number of cores", value=8, min_value=1)
    update_every_evaluations = st.number_input(
        "Actualizar cada N evaluaciones",
        value=DEFAULT_UPDATE_EVERY_EVALUATIONS,
        min_value=1,
        step=100,
    )

    default_base_text = parameter_space_text(
        evolver_home, algorithm.encodings[algorithm.runnable_encoding]
    )
    parameter_space_text_value = _render_parameter_space_editor(
        f"Espacio de parámetros del algoritmo base ({algorithm.name})",
        default_base_text,
        f"base_{algorithm.name}",
    )

    with st.expander(f"Operator flags del meta-optimizador ({LAUNCHABLE_META_ALGORITHM_NAME})"):
        operator_flags = _render_operator_flags_editor(
            _default_operator_flags_text(evolver_home), "meta_operator_flags"
        )

    base_level = _base_level_config(
        algorithm.registry_name, "<written to disk at launch>", extra_config
    )
    meta_search = _meta_search_config(
        int(meta_max_evaluations), int(number_of_cores), operator_flags or {}
    )
    st.code(base_level_to_yaml(base_level), language="yaml")
    st.code(flat_meta_search_to_yaml(meta_search), language="yaml")

    can_launch = parameter_space_text_value is not None and operator_flags is not None
    if st.button("Ejecutar entrenamiento", disabled=not can_launch):
        run_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        _launch_run(
            evolver_home,
            run_id,
            algorithm.registry_name,
            output_directory_base,
            parameter_space_text_value,
            extra_config,
            operator_flags,
            int(meta_max_evaluations),
            int(number_of_cores),
            int(update_every_evaluations),
        )
        st.rerun()

    _render_last_finished_run_if_any(evolver_home)
