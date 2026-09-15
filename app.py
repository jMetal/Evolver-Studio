"""Streamlit prototype: run Evolver's ZDT4 training and show its results."""

import datetime as dt
import time
from pathlib import Path

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from evolver_studio.adaptive_poll import AdaptivePollInterval
from evolver_studio.evolver_client import (
    RunState,
    RunStatus,
    build_jar,
    jar_path,
    read_status,
    start_training,
)
from evolver_studio.live_front import LiveFrontRenderer, build_front_figure
from evolver_studio.request import BaseLevelConfig, FlatMetaSearchConfig, to_request_yaml
from evolver_studio.result import Err
from evolver_studio.results import (
    checkpoint_front,
    latest_checkpoint_evaluation,
    list_output_dir,
    load_indicators,
    read_metadata,
    read_results_pointer,
)

DEFAULT_EVOLVER_HOME = "/Users/ajnebro/Softw/Evolver"


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


def _poll_tick(
    status_yaml: Path,
    indicators_csv: Path,
    progress: DeltaGenerator,
    chart_placeholder: DeltaGenerator,
    renderer: LiveFrontRenderer,
    interval: AdaptivePollInterval,
) -> RunStatus | None:
    """Read the latest status, redraw the live front if due, and adapt the poll interval.

    Args:
        status_yaml: Path to the run's status file.
        indicators_csv: Path to the run's (still-growing) INDICATORS.csv.
        progress: Progress bar to update with the latest evaluation count.
        chart_placeholder: Placeholder to redraw the live front into.
        renderer: Tracks render-throttling state across polls.
        interval: Tracks the adaptive sleep interval across polls.

    Returns:
        The latest parsed status, or None if not yet available.
    """
    status = read_status(status_yaml)
    if status is not None:
        fraction = status.evaluations_done / max(status.max_evaluations, 1)
        progress.progress(fraction, text=f"{status.evaluations_done}/{status.max_evaluations}")
    update = renderer.poll(indicators_csv)
    interval.record_poll(update.changed, time.monotonic())
    if update.figure is not None:
        chart_placeholder.plotly_chart(
            update.figure,
            use_container_width=True,
            key=f"live_indicator_front_{update.evaluation}",
        )
    return status


def _run_and_wait(
    evolver_home: Path,
    jar: Path,
    request_yaml: Path,
    status_yaml: Path,
    indicators_csv: Path,
    update_every_evaluations: int,
) -> RunStatus:
    """Launch training and poll status.yaml, refreshing a live front preview.

    The poll interval self-adjusts (see AdaptivePollInterval): it isn't meant
    to keep the preview in tight real-time sync with INDICATORS.csv, only to
    avoid wasted polls while backing off, and to catch up once data appears.

    Args:
        evolver_home: Path to the Evolver checkout (JVM working directory).
        jar: Path to Evolver's fat jar.
        request_yaml: Path to the written training request.
        status_yaml: Path where progress is reported.
        indicators_csv: Path where checkpoints of the indicator front are appended.
        update_every_evaluations: Minimum evaluations between live-preview redraws.

    Returns:
        The final run status (FINISHED or FAILED).
    """
    start_training(evolver_home, jar, request_yaml, status_yaml)
    progress = st.progress(0.0, text="Starting…")
    chart_placeholder = st.empty()
    renderer = LiveFrontRenderer(update_every_evaluations)
    interval = AdaptivePollInterval()
    status = _poll_tick(
        status_yaml, indicators_csv, progress, chart_placeholder, renderer, interval
    )
    while status is None or status.state == RunState.RUNNING:
        time.sleep(interval.seconds())
        status = _poll_tick(
            status_yaml, indicators_csv, progress, chart_placeholder, renderer, interval
        )
    return status


def _render_indicator_front(indicators_csv: Path) -> None:
    """Plot the training's final non-dominated indicator front.

    Args:
        indicators_csv: Path to the completed run's INDICATORS.csv.
    """
    history = load_indicators(indicators_csv)
    latest = latest_checkpoint_evaluation(history)
    if latest is None:
        st.info("No indicator data was written.")
        return
    st.plotly_chart(
        build_front_figure(checkpoint_front(history, latest), latest), key="final_indicator_front"
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


st.title("Evolver-Studio — ZDT4 training prototype")

evolver_home = Path(st.sidebar.text_input("Evolver checkout path", DEFAULT_EVOLVER_HOME))
if st.sidebar.button("Compilar Evolver"):
    build_result = build_jar(evolver_home)
    if isinstance(build_result, Err):
        st.sidebar.error(build_result.message)
    else:
        st.sidebar.success("Jar built successfully.")

output_directory_base = st.text_input("Output directory", "results/nsgaii/ZDT4")
meta_max_evaluations = st.number_input("Meta max evaluations", value=2000, min_value=100)
number_of_cores = st.number_input("Number of cores", value=8, min_value=1)
update_every_evaluations = st.number_input(
    "Actualizar cada N evaluaciones", value=100, min_value=1, step=100
)

base_level = _base_level_config(output_directory_base)
meta_search = _meta_search_config(int(meta_max_evaluations), int(number_of_cores))
st.code(to_request_yaml(base_level, meta_search), language="yaml")

if st.button("Ejecutar entrenamiento"):
    run_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = evolver_home / "cli-runner-runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    request_yaml = run_dir / "request.yaml"
    status_yaml = run_dir / "status.yaml"

    # Nest under run_id so INDICATORS.csv (append-only in Evolver) never mixes
    # checkpoints across runs that share the same output_directory_base.
    run_base_level = _base_level_config(f"{output_directory_base}/{run_id}")
    request_yaml.write_text(to_request_yaml(run_base_level, meta_search))
    indicators_csv = evolver_home / run_base_level.output_directory / "INDICATORS.csv"

    final_status = _run_and_wait(
        evolver_home,
        jar_path(evolver_home),
        request_yaml,
        status_yaml,
        indicators_csv,
        int(update_every_evaluations),
    )
    if final_status.state == RunState.FAILED:
        st.error(final_status.error_message)
    else:
        results_yaml = run_dir / "results.yaml"
        pointer = read_results_pointer(results_yaml, evolver_home)
        st.success("Training finished.")
        if st.checkbox("Mostrar frente de indicadores", value=True):
            _render_indicator_front(pointer.indicators_file)
        _render_output_directory(pointer.output_directory, pointer.metadata_file)
