"""Streamlit prototype: run Evolver's ZDT4 training and show its results."""

import datetime as dt
import time
from pathlib import Path

import plotly.express as px
import streamlit as st

from evolver_studio.evolver_client import (
    RunState,
    RunStatus,
    build_jar,
    jar_path,
    read_status,
    start_training,
)
from evolver_studio.request import BaseLevelConfig, FlatMetaSearchConfig, to_request_yaml
from evolver_studio.result import Err
from evolver_studio.results import (
    list_output_dir,
    load_indicators,
    read_metadata,
    read_results_pointer,
)

DEFAULT_EVOLVER_HOME = "/Users/ajnebro/Softw/Evolver"
POLL_INTERVAL_SECONDS = 1.0


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


def _run_and_wait(
    evolver_home: Path, jar: Path, request_yaml: Path, status_yaml: Path
) -> RunStatus:
    """Launch training and poll status.yaml until it reaches a terminal state.

    Args:
        evolver_home: Path to the Evolver checkout (JVM working directory).
        jar: Path to Evolver's fat jar.
        request_yaml: Path to the written training request.
        status_yaml: Path where progress is reported.

    Returns:
        The final run status (FINISHED or FAILED).
    """
    start_training(evolver_home, jar, request_yaml, status_yaml)
    progress = st.progress(0.0, text="Starting…")
    status = read_status(status_yaml)
    while status is None or status.state == RunState.RUNNING:
        time.sleep(POLL_INTERVAL_SECONDS)
        status = read_status(status_yaml)
        if status is not None:
            fraction = status.evaluations_done / max(status.max_evaluations, 1)
            progress.progress(fraction, text=f"{status.evaluations_done}/{status.max_evaluations}")
    return status


def _render_indicator_front(indicators_csv: Path) -> None:
    """Plot the meta-level non-dominated archive as a scatter of two indicators.

    INDICATORS.csv's indicator columns are named after jMetal's short indicator
    codes (e.g. "EP", "NHV"), not the full names given in the request — so the
    axes are taken from the CSV header itself rather than from indicator_names.

    Args:
        indicators_csv: Path to INDICATORS.csv, with columns
            Evaluation, SolutionId, <indicator1>, <indicator2>, ...
    """
    indicators = load_indicators(indicators_csv)
    x_axis, y_axis = indicators.columns[2], indicators.columns[3]
    st.plotly_chart(px.scatter(indicators, x=x_axis, y=y_axis, title="Meta-level indicator front"))


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

output_directory = st.text_input("Output directory", "results/nsgaii/ZDT4")
meta_max_evaluations = st.number_input("Meta max evaluations", value=2000, min_value=100)
number_of_cores = st.number_input("Number of cores", value=8, min_value=1)

base_level = _base_level_config(output_directory)
meta_search = _meta_search_config(int(meta_max_evaluations), int(number_of_cores))
st.code(to_request_yaml(base_level, meta_search), language="yaml")

if st.button("Ejecutar entrenamiento"):
    run_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = evolver_home / "cli-runner-runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    request_yaml = run_dir / "request.yaml"
    status_yaml = run_dir / "status.yaml"
    request_yaml.write_text(to_request_yaml(base_level, meta_search))

    final_status = _run_and_wait(evolver_home, jar_path(evolver_home), request_yaml, status_yaml)
    if final_status.state == RunState.FAILED:
        st.error(final_status.error_message)
    else:
        results_yaml = run_dir / "results.yaml"
        pointer = read_results_pointer(results_yaml, evolver_home)
        st.success("Training finished.")
        if st.checkbox("Mostrar frente de indicadores", value=True):
            _render_indicator_front(pointer.indicators_file)
        _render_output_directory(pointer.output_directory, pointer.metadata_file)
