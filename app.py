"""Streamlit prototype: run Evolver's ZDT4 training and show its results.

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

import streamlit as st

from evolver_studio.adaptive_poll import AdaptivePollInterval
from evolver_studio.evolver_client import (
    RunState,
    RunStatus,
    build_jar,
    cancel_training,
    jar_path,
    read_pid,
    read_status,
    start_training,
    write_pid_file,
)
from evolver_studio.live_front import LiveFrontRenderer, build_front_figure
from evolver_studio.request import BaseLevelConfig, FlatMetaSearchConfig, to_request_yaml
from evolver_studio.result import Err
from evolver_studio.results import (
    deduplicate_consecutive_checkpoints,
    list_output_dir,
    load_indicators,
    read_metadata,
    read_results_pointer,
)
from evolver_studio.runs import ActiveRun, find_active_run, mark_cancelled

DEFAULT_EVOLVER_HOME = "/Users/ajnebro/Softw/Evolver"
DEFAULT_UPDATE_EVERY_EVALUATIONS = 100
LIVE_FRAGMENT_RUN_EVERY_SECONDS = 2


def _base_level_config(output_directory: str) -> BaseLevelConfig:
    """Build the fixed ZDT4 base-level config, matching Zdt4TrainingRunner.java.

    Args:
        output_directory: Where Evolver writes this run's results.

    Returns:
        The base-level config for NSGA-II tuned on ZDT4.
    """
    return BaseLevelConfig(
        algorithm_name="NSGA-II",
        population_size=100,
        number_of_independent_runs=1,
        yaml_parameter_space_file="NSGAIIDouble.yaml",
        extra_config=None,
        training_problem_names=["ZDT4"],
        training_reference_front_file_names=["resources/referenceFronts/ZDT4.csv"],
        training_evaluations=[12000],
        indicator_names=["Epsilon", "NormalizedHypervolume"],
        output_directory=output_directory,
    )


def _meta_search_config(meta_max_evaluations: int, number_of_cores: int) -> FlatMetaSearchConfig:
    """Build the flat meta-search config, matching Zdt4TrainingRunner.java.

    Args:
        meta_max_evaluations: Meta-level evaluation budget.
        number_of_cores: Cores used to parallelize base-level runs.

    Returns:
        The flat meta-search config for the meta-level NSGA-II.
    """
    return FlatMetaSearchConfig(
        meta_max_evaluations=meta_max_evaluations,
        meta_population_size=100,
        number_of_cores=number_of_cores,
        mutation_probability_factor=1.5,
        meta_yaml_parameter_space_file="NSGAIIDoubleReduced.yaml",
    )


def _launch_run(
    evolver_home: Path,
    run_id: str,
    base_level: BaseLevelConfig,
    meta_search: FlatMetaSearchConfig,
    update_every_evaluations: int,
) -> None:
    """Write the request, launch training, and persist its PID for later control.

    Args:
        evolver_home: Path to the Evolver checkout (JVM working directory).
        run_id: This run's timestamp-based identifier.
        base_level: Base-level config, with output_directory nested under run_id
            so INDICATORS.csv (append-only in Evolver) never mixes checkpoints
            across runs that share the same base output directory.
        meta_search: Meta-search config.
        update_every_evaluations: Chosen live-preview redraw threshold.
    """
    run_dir = evolver_home / "cli-runner-runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    request_yaml = run_dir / "request.yaml"
    request_yaml.write_text(to_request_yaml(base_level, meta_search))
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


def _poll_tick(
    status_yaml: Path,
    indicators_csv: Path,
    renderer: LiveFrontRenderer,
    interval: AdaptivePollInterval,
) -> RunStatus | None:
    """Read the latest status, redraw the live front, and adapt the poll interval.

    Draws directly into whatever container is active when called — meant to
    run inside a fragment, which fully replaces its own contents each time it
    reruns, so both the progress bar and the front preview (last known figure,
    not just a newly-due one, to avoid it flickering away between throttled
    redraws) are (re)drawn unconditionally on every call.

    Args:
        status_yaml: Path to the run's status file.
        indicators_csv: Path to the run's (still-growing) INDICATORS.csv.
        renderer: Tracks render-throttling state across polls.
        interval: Tracks the adaptive sleep interval across polls.

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
    if renderer.last_figure is not None:
        st.plotly_chart(
            renderer.last_figure,
            use_container_width=True,
            key=f"live_indicator_front_{renderer.last_rendered_evaluation}",
        )
    return status


def _render_indicator_front(indicators_csv: Path) -> None:
    """Plot the indicator front's evolution across all checkpoints written.

    Args:
        indicators_csv: Path to the completed run's INDICATORS.csv.
    """
    history = load_indicators(indicators_csv)
    if history.empty:
        st.info("No indicator data was written.")
        return
    figure = build_front_figure(deduplicate_consecutive_checkpoints(history))
    st.plotly_chart(figure, key="final_indicator_front")


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
        _render_indicator_front(pointer.indicators_file)
    _render_output_directory(pointer.output_directory, pointer.metadata_file)


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
        status = _poll_tick(active_run.status_yaml, active_run.indicators_csv, renderer, interval)
        if status is not None and status.state != RunState.RUNNING:
            st.session_state["last_finished_run"] = (status, active_run.run_dir)
            st.rerun()

    _poll()


st.title("Evolver-Studio — ZDT4 training prototype")

evolver_home = Path(st.sidebar.text_input("Evolver checkout path", DEFAULT_EVOLVER_HOME))
if st.sidebar.button("Compilar Evolver"):
    build_result = build_jar(evolver_home)
    if isinstance(build_result, Err):
        st.sidebar.error(build_result.message)
    else:
        st.sidebar.success("Jar built successfully.")

active_run = find_active_run(evolver_home)

if active_run is not None:
    _render_active_run(evolver_home, active_run)
else:
    output_directory_base = st.text_input("Output directory", "results/nsgaii/ZDT4")
    meta_max_evaluations = st.number_input("Meta max evaluations", value=2000, min_value=100)
    number_of_cores = st.number_input("Number of cores", value=8, min_value=1)
    update_every_evaluations = st.number_input(
        "Actualizar cada N evaluaciones",
        value=DEFAULT_UPDATE_EVERY_EVALUATIONS,
        min_value=1,
        step=100,
    )

    base_level = _base_level_config(output_directory_base)
    meta_search = _meta_search_config(int(meta_max_evaluations), int(number_of_cores))
    st.code(to_request_yaml(base_level, meta_search), language="yaml")

    if st.button("Ejecutar entrenamiento"):
        run_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        run_base_level = _base_level_config(f"{output_directory_base}/{run_id}")
        _launch_run(
            evolver_home, run_id, run_base_level, meta_search, int(update_every_evaluations)
        )
        st.rerun()

    _render_last_finished_run_if_any(evolver_home)
