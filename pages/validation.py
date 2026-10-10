"""Validation: compare a tuned configuration with other algorithms on a set of problems.

The page builds a validation study (see evolver_studio/validation.py): the tuned configuration,
which is the pivot, and the default configurations of other Evolver algorithms, all run many times
on the same problems with the same population size, evaluation budget and seeds. A detached worker
runs it, and its results are compared with medians, a Wilcoxon test and the A12 effect size.
"""

import datetime as dt
import os
import zlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from evolver_studio import running_badge
from evolver_studio.app_state import (
    registered_problems,
    require_evolver_jar,
    warn_if_jar_older_than_catalogue,
)
from evolver_studio.catalogue import BASE_ALGORITHMS, QUALITY_INDICATORS
from evolver_studio.configuration import (
    complete_values,
    configuration_string,
    modified_values,
    parse_configuration,
    values_outside_the_space,
)
from evolver_studio.evolver_client import WORKING_DIRECTORY, read_pid, read_status
from evolver_studio.figure_export import render_chart
from evolver_studio.parameter_form import render_configuration_form
from evolver_studio.parameter_space import parse_parameter_space
from evolver_studio.problem_browser import render_problem_selector
from evolver_studio.problem_catalogue import Problem, problems_with_encoding
from evolver_studio.progress import study_running_label
from evolver_studio.resource_files import default_configuration_text, parameter_space_text
from evolver_studio.runs import RunPhase, find_run_in_progress, mark_cancelled, run_phase
from evolver_studio.saes_export import (
    saes_available,
    saes_metrics,
    saes_results,
    wilcoxon_pivot_table,
)
from evolver_studio.training_runs import FinishedTraining, list_finished_trainings
from evolver_studio.validation import (
    DEFAULT_MAX_EVALUATIONS,
    DEFAULT_POPULATION_SIZE,
    DEFAULT_RUNS,
    Contender,
    StudyInfo,
    StudyProblem,
    ValidationStudy,
    base_algorithm,
    collect_runs,
    default_contenders,
    list_studies,
    plan_jobs,
    write_study,
)
from evolver_studio.validation_form import (
    ENCODING_KEY,
    EVALUATIONS_KEY,
    INDICATORS_KEY,
    PASTED_ALGORITHM_KEY,
    PASTED_CONFIGURATION_KEY,
    POPULATION_KEY,
    PROBLEM_COLUMNS,
    PROBLEM_ROWS_KEY,
    RUNS_KEY,
    SEED_KEY,
    TUNED_FROM_TRAINING,
    TUNED_PASTED,
    TUNED_SOURCE_KEY,
    adjusted_key,
    adjusted_name,
    configuration_errors,
    defaults_key,
    parse_problem_table,
    problem_table,
    problems_key,
    tuned_name_key,
)
from evolver_studio.validation_runner import (
    VALIDATION_RUNS_DIRECTORY_NAME,
    cancel_study,
    start_study,
)
from evolver_studio.validation_stats import (
    ALPHA,
    best_contenders,
    compare_with_pivot,
    indicator_names,
    interquartile_ranges,
    medians,
    verdict_counts,
)

RUNS_DIRECTORY = WORKING_DIRECTORY / VALIDATION_RUNS_DIRECTORY_NAME
POLL_EVERY_SECONDS = 1
DEFAULT_INDICATORS = ("Epsilon", "NormalizedHypervolume")
ENCODINGS = ("Double", "Binary", "Permutation")
HISTORY_KEY = "validation_history"
SPACE_COLUMN_HELP = (
    "Optional: the constructor's arguments, all of them or none, comma-separated "
    "(e.g. 12, 2 for DTLZ2 with 12 variables and 2 objectives). Explore › Problems lists them."
)


@dataclass(slots=True, frozen=True)
class TunedChoice:
    """What the Tuned section chooses.

    Attributes:
        contender: The tuned configuration as a contender, or None while it is not complete.
        errors: What is wrong with it.
    """

    contender: Contender | None
    errors: list[str]


def _render_problems(catalogue: dict[str, Problem]) -> tuple[str, tuple[StudyProblem, ...]]:
    """Section 1: the encoding and the problems, each with its arguments and reference front.

    Args:
        catalogue: The problem catalogue.

    Returns:
        The encoding and the problems (empty while any row of the table is wrong).
    """
    st.subheader("1. Problems")
    encoding = st.selectbox(
        "Encoding of the problems",
        ENCODINGS,
        key=ENCODING_KEY,
        help="All the problems of a study share it: an algorithm solves the problems of the "
        "encodings it supports, so the algorithms to compare are those of this encoding.",
    )
    names = render_problem_selector(
        catalogue,
        problems_with_encoding(catalogue, encoding),
        encoding,
        "Problems",
        problems_key(encoding),
        "validation",
    )
    if not names:
        return encoding, ()
    table = problem_table(
        names, catalogue, WORKING_DIRECTORY, st.session_state.get(PROBLEM_ROWS_KEY)
    )
    edited = st.data_editor(
        table,
        column_config={
            "problem": st.column_config.TextColumn("Problem", disabled=True),
            "arguments": st.column_config.TextColumn("Arguments", help=SPACE_COLUMN_HELP),
            "reference_front": st.column_config.TextColumn(
                "Reference front file",
                help="The indicators compare each front with it; fronts with a dimension suffix "
                "(3D) belong to the problem with that number of objectives.",
            ),
        },
        hide_index=True,
        width="stretch",
        key=f"validation_problem_table_{encoding}_{'|'.join(names)}",
    )
    st.session_state[PROBLEM_ROWS_KEY] = edited[list(PROBLEM_COLUMNS)]
    problems, errors = parse_problem_table(edited, catalogue, encoding, WORKING_DIRECTORY)
    for error in errors:
        st.error(error)
    return encoding, () if errors else problems


def _tuned_from_training(encoding: str) -> tuple[str, str] | None:
    """Choose a configuration a finished training run found.

    Returns:
        The algorithm's registry name and the configuration, or None while none is chosen.
    """
    trainings = [t for t in list_finished_trainings(WORKING_DIRECTORY) if t.encoding == encoding]
    if not trainings:
        st.info(f"No finished training run for {encoding} problems: paste a configuration.")
        return None
    training: FinishedTraining = st.selectbox(
        "Training run",
        trainings,
        format_func=lambda t: t.label,
        key="validation_training",
    )
    index = st.selectbox(
        "Configuration of its final front",
        range(len(training.configurations)),
        format_func=lambda i: f"{i + 1}. {training.configurations[i].label}",
        key=f"validation_configuration_{training.run_id}",
        help="The configurations the training ended with, each with its values of the "
        "meta-objectives. A validation shows how good one of them really is.",
    )
    return training.algorithm, training.configurations[index].configuration


def _tuned_pasted(encoding: str) -> tuple[str, str] | None:
    """Paste a configuration, and choose the algorithm it is for.

    Returns:
        The algorithm's registry name and the configuration, or None while it is empty.
    """
    algorithms = [
        a for a in BASE_ALGORITHMS if a.runnable_today and encoding in a.runnable_encodings
    ]
    name = st.selectbox("Algorithm", [a.name for a in algorithms], key=PASTED_ALGORITHM_KEY)
    configuration = st.text_area(
        "Configuration",
        placeholder="--algorithmResult population --crossover SBX ...",
        key=PASTED_CONFIGURATION_KEY,
        help="As a training run writes it: --parameter value pairs.",
    )
    if not configuration.strip():
        return None
    algorithm = next(a for a in algorithms if a.name == name)
    return algorithm.registry_name, configuration.strip()


def _render_tuned(encoding: str, jar: Path) -> TunedChoice:
    """Section 2a: the tuned configuration, the pivot every other contender is compared with."""
    source = st.radio(
        "Tuned configuration from",
        (TUNED_FROM_TRAINING, TUNED_PASTED),
        horizontal=True,
        key=TUNED_SOURCE_KEY,
    )
    chosen = (
        _tuned_from_training(encoding) if source == TUNED_FROM_TRAINING else _tuned_pasted(encoding)
    )
    if chosen is None:
        return TunedChoice(None, [])
    registry_name, configuration = chosen
    algorithm = base_algorithm(registry_name)
    assert algorithm is not None
    name = st.text_input(
        "Name in the tables",
        f"{algorithm.name} (tuned)",
        key=tuned_name_key(registry_name),
    )
    errors = configuration_errors(
        configuration, parameter_space_text(jar, algorithm.encodings[encoding]), name
    )
    for error in errors:
        st.error(error)
    st.code(configuration, language=None, wrap_lines=True)
    return TunedChoice(Contender(name.strip(), registry_name, configuration), errors)


def _render_contenders(
    encoding: str, jar: Path
) -> tuple[tuple[Contender, ...], str | None, list[str]]:
    """Section 2: the tuned configuration (the pivot) and the algorithms to compare it with.

    Returns:
        The contenders, the pivot's name (None while there is none), and what is wrong.
    """
    st.subheader("2. Contenders")
    tuned = _render_tuned(encoding, jar)
    defaults = default_contenders(
        encoding, lambda file_name: default_configuration_text(jar, file_name)
    )
    chosen = st.multiselect(
        "Algorithms to compare it with",
        [c.name for c in defaults],
        default=["NSGA-II"] if any(c.name == "NSGA-II" for c in defaults) else [],
        key=defaults_key(encoding),
        help="Every algorithm of Evolver that has a default configuration for this encoding. "
        "Each starts from it; adjust its parameters below if you want.",
    )
    others = tuple(_adjust(c, encoding, jar) for c in defaults if c.name in chosen)
    if tuned.contender is None:
        return others, None, tuned.errors
    return (tuned.contender, *others), tuned.contender.name, tuned.errors


def _adjust(contender: Contender, encoding: str, jar: Path) -> Contender:
    """Let the user adjust the default configuration of an algorithm compared with.

    The form of Run algorithm, within the algorithm's parameter space; the marks show what
    differs from the default configuration.

    Args:
        contender: The algorithm with its default configuration.
        encoding: The encoding of the problems.
        jar: Path to Evolver's jar, which holds the parameter space.

    Returns:
        The contender as it is, or with the configuration adjusted and its name marked so.
    """
    algorithm = base_algorithm(contender.algorithm)
    if algorithm is None or encoding not in algorithm.encodings:
        return contender
    parameters = parse_parameter_space(parameter_space_text(jar, algorithm.encodings[encoding]))
    reference = parse_configuration(contender.configuration)
    start_key = adjusted_key(encoding, contender.name)
    start = st.session_state.get(start_key) or reference
    values = complete_values(parameters, start)
    version_key = f"{start_key}_version"
    version = st.session_state.setdefault(version_key, 0)
    # A new start (a study chosen again) gives new widgets, which then show it.
    fingerprint = zlib.crc32(repr(sorted(start.items())).encode())
    key_prefix = f"{start_key}_{version}_{fingerprint:x}"
    with st.expander(f"Adjust the parameters of {contender.name}"):
        if st.button("Reset to the default configuration", key=f"{start_key}_reset"):
            st.session_state.pop(start_key, None)
            st.session_state[version_key] = version + 1
            st.rerun()
        edited = render_configuration_form(parameters, values, reference, key_prefix)
        outside = values_outside_the_space(parameters, edited)
        if outside:
            st.warning("Outside the parameter space: " + ", ".join(f"`{name}`" for name in outside))
    changed = modified_values(parameters, edited, reference)
    if not changed:
        return contender
    st.caption(
        f"**{adjusted_name(contender.name)}**: {len(changed)} parameter(s) changed from the "
        f"default configuration ({', '.join(f'`{name}`' for name in changed)})."
    )
    return Contender(
        adjusted_name(contender.name), contender.algorithm, configuration_string(parameters, edited)
    )


@dataclass(slots=True, frozen=True)
class Budget:
    """What section 3 chooses."""

    population_size: int
    max_evaluations: int
    runs: int
    seed: int
    indicators: tuple[str, ...]
    processes: int


def _render_budget() -> Budget:
    """Section 3: what every contender runs with."""
    st.subheader("3. Budget")
    columns = st.columns(5)
    population = columns[0].number_input(
        "Population size",
        2,
        value=DEFAULT_POPULATION_SIZE,
        key=POPULATION_KEY,
        help="The same for every algorithm. MOEA/D, RVEA and NSGA-III need a weight vector file "
        "that matches it and the problem's number of objectives; they are left out of the "
        "problems that have none.",
    )
    evaluations = columns[1].number_input(
        "Evaluations per run",
        100,
        value=DEFAULT_MAX_EVALUATIONS,
        step=1000,
        key=EVALUATIONS_KEY,
    )
    runs = columns[2].number_input(
        "Independent runs",
        2,
        value=DEFAULT_RUNS,
        key=RUNS_KEY,
        help="Per algorithm and problem. Run i uses the seed + i - 1, the same for every "
        "algorithm.",
    )
    seed = columns[3].number_input("First seed", 0, value=1, key=SEED_KEY)
    processes = columns[4].number_input(
        "Parallel jobs",
        1,
        value=min(os.cpu_count() or 1, 8),
        key="validation_processes",
        help="How many algorithm-problem pairs run at the same time, each in its own JVM.",
    )
    indicators = st.multiselect(
        "Quality indicators",
        [indicator.registry_name for indicator in QUALITY_INDICATORS],
        default=list(DEFAULT_INDICATORS),
        key=INDICATORS_KEY,
        help="Computed on each run's front, normalized with the problem's reference front. "
        "All are minimized.",
    )
    return Budget(
        int(population), int(evaluations), int(runs), int(seed), tuple(indicators), int(processes)
    )


def _launch(study: ValidationStudy, processes: int, jar: Path) -> None:
    """Write the study and start it."""
    study_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    directory = RUNS_DIRECTORY / study_id
    write_study(study, directory, WORKING_DIRECTORY)
    start_study(directory, jar, processes, WORKING_DIRECTORY)
    st.session_state.pop(HISTORY_KEY, None)


def _render_form(catalogue: dict[str, Problem], jar: Path, can_run: bool) -> None:
    """The three sections that build a study, and the button that launches it."""
    encoding, problems = _render_problems(catalogue)
    contenders, pivot, errors = _render_contenders(encoding, jar)
    budget = _render_budget()
    study = ValidationStudy(
        encoding=encoding,
        problems=problems,
        contenders=contenders,
        pivot=pivot or "",
        population_size=budget.population_size,
        max_evaluations=budget.max_evaluations,
        runs=budget.runs,
        seed=budget.seed,
        indicators=budget.indicators,
    )
    errors = [*errors, *study.validation_errors()]
    jobs, skipped = (
        plan_jobs(study, RUNS_DIRECTORY / "preview", WORKING_DIRECTORY) if not errors else ([], [])
    )
    for error in errors:
        st.warning(error)
    for job in skipped:
        st.warning(f"{job.contender} is left out of {job.problem}: {job.reason}.")
    if jobs:
        st.caption(
            f"{len(jobs)} jobs of {study.runs} runs each, {len(jobs) * study.runs} runs in all, "
            f"with {study.max_evaluations:,} evaluations each."
        )
    if st.button("Run the study", type="primary", disabled=bool(errors) or not can_run):
        _launch(study, budget.processes, jar)
        st.rerun()


def _cancel(study_directory: Path) -> None:
    pid = read_pid(study_directory / "pid.txt")
    if pid is not None:
        cancel_study(pid)
    mark_cancelled(study_directory)


def _render_study_in_progress(study_directory: Path) -> None:
    """Show that a study is in progress, how many jobs are done, and move on to its results."""
    running_badge.inject_style()

    @st.fragment(run_every=POLL_EVERY_SECONDS, key=f"validation_poll_{study_directory.name}")
    def _poll() -> None:
        running_badge.render(study_running_label(read_status(study_directory / "status.yaml")))
        if run_phase(study_directory) not in (RunPhase.STARTING, RunPhase.RUNNING):
            st.session_state[HISTORY_KEY] = study_directory.name
            st.rerun()

    _poll()
    if st.button("Cancel the study"):
        _cancel(study_directory)
        st.rerun()


def _failed_jobs(study_directory: Path) -> list[int]:
    try:
        return [int(n) for n in (study_directory / "failed_jobs.txt").read_text().split()]
    except (OSError, ValueError):
        return []


def _render_summary(runs: pd.DataFrame, indicator: str, pivot: str) -> None:
    table = medians(runs, indicator)
    best = best_contenders(table)
    st.markdown(
        f"**Median of {indicator}** over the runs; the lowest on each problem is highlighted."
    )
    st.dataframe(table.style.highlight_min(axis=1).format("{:.4g}"), width="stretch")
    st.caption(
        "Best on "
        + ", ".join(
            f"{name}: {(best == name).sum()}" for name in table.columns if (best == name).sum()
        )
        + f" of {len(table)} problems."
    )
    with st.expander("Interquartile range"):
        st.dataframe(interquartile_ranges(runs, indicator).style.format("{:.4g}"), width="stretch")


def _render_comparison(runs: pd.DataFrame, indicator: str, pivot: str) -> None:
    comparison = compare_with_pivot(runs, indicator, pivot)
    if comparison.empty:
        st.info("There is nothing to compare yet.")
        return
    st.markdown(
        f"**{pivot}** against each other algorithm, problem by problem, on {indicator} (lower is "
        f"better). A Wilcoxon rank-sum test at {ALPHA}, and the **A12** effect size: the "
        "probability that a run of the pivot is better than a run of the other algorithm "
        "(0.5 is no difference; 1 means always better)."
    )
    st.markdown("On how many problems the pivot is better, not different, or worse:")
    st.dataframe(verdict_counts(comparison), width="stretch")
    detail = comparison.rename(
        columns={
            "problem": "Problem",
            "contender": "Algorithm",
            "pivot_median": "Pivot median",
            "median": "Median",
            "p_value": "p-value",
            "a12": "A12",
            "magnitude": "Effect",
            "verdict": "Verdict",
        }
    )
    st.dataframe(detail, hide_index=True, width="stretch")
    st.download_button(
        "Download the comparison (CSV)",
        detail.to_csv(index=False),
        file_name=f"comparison_{indicator}.csv",
        mime="text/csv",
        key=f"validation_comparison_{indicator}",
    )
    if saes_available():
        st.download_button(
            "Download the Wilcoxon pivot table (LaTeX)",
            lambda: wilcoxon_pivot_table(runs, indicator, pivot),
            file_name=f"WilcoxonPivot_{indicator}.tex",
            mime="application/x-tex",
            key=f"validation_wilcoxon_latex_{indicator}",
            on_click="ignore",
            icon=":material/download:",
            help="SAES's Wilcoxon pivot table, as Evolver's scripts/wilcoxon_pivot_tables.py "
            "makes it: median and interquartile range, the pivot in the last column, + the pivot "
            "is significantly better, - significantly worse, = no significant difference, and "
            "the last row counts them. A LaTeX document that compiles on its own.",
        )
    else:
        st.caption(
            "Install SAES (`pip install SAES`) to download this comparison as SAES's Wilcoxon "
            "pivot table in LaTeX."
        )


def _render_boxplots(runs: pd.DataFrame, indicator: str) -> None:
    figure = px.box(
        runs,
        x="contender",
        y=indicator,
        color="contender",
        facet_col="problem",
        facet_col_wrap=3,
        labels={"contender": "", indicator: indicator},
    )
    figure.update_yaxes(matches=None, showticklabels=True)
    figure.update_xaxes(showticklabels=False)
    figure.update_layout(showlegend=True, height=320 * -(-runs["problem"].nunique() // 3))
    render_chart(
        figure, f"validation_boxplot_{indicator}", f"boxplots_{indicator}", width="stretch"
    )


def _render_details(study_directory: Path, manifest: dict) -> None:
    st.markdown(
        f"**Budget**: population {manifest['populationSize']}, {manifest['maxEvaluations']:,} "
        f"evaluations, {manifest['runs']} runs from the seed {manifest['seed']}, "
        f"{manifest['encoding']} problems. Folder: `{study_directory}`."
    )
    st.dataframe(pd.DataFrame(manifest["contenders"]), hide_index=True, width="stretch")
    if manifest.get("skipped"):
        st.warning(
            "Left out: "
            + "; ".join(
                f"{s['contender']} on {s['problem']} ({s['reason']})" for s in manifest["skipped"]
            )
        )
    failed = _failed_jobs(study_directory)
    if failed:
        st.error(
            f"Jobs that failed: {', '.join(map(str, failed))}. See `runner.log` in their folders."
        )
    st.download_button(
        "Download study.yaml",
        (study_directory / "study.yaml").read_text(),
        file_name="study.yaml",
        mime="application/x-yaml",
        key=f"validation_manifest_{study_directory.name}",
    )


def _render_results(study: StudyInfo) -> None:
    """Show the results of a study: its summary, comparison, boxplots and runs."""
    phase = run_phase(study.directory)
    if phase == RunPhase.CANCELLED:
        st.info("This study was cancelled: what follows is what had finished.")
    elif phase in (RunPhase.STARTING, RunPhase.RUNNING):
        st.info("This study is still in progress.")
    runs = collect_runs(study.directory, WORKING_DIRECTORY)
    if runs.empty:
        st.info("No job has finished yet.")
        return
    indicators = indicator_names(runs)
    indicator = st.selectbox("Indicator", indicators, key=f"validation_indicator_{study.study_id}")
    tabs = st.tabs(["Summary", "Comparison", "Boxplots", "Runs", "Details"])
    with tabs[0]:
        _render_summary(runs, indicator, study.manifest["pivot"])
    with tabs[1]:
        _render_comparison(runs, indicator, study.manifest["pivot"])
    with tabs[2]:
        _render_boxplots(runs, indicator)
    with tabs[3]:
        st.dataframe(runs, hide_index=True, width="stretch")
        st.download_button(
            "Download every run (CSV)",
            runs.to_csv(index=False),
            file_name=f"{study.study_id}_runs.csv",
            mime="text/csv",
            key=f"validation_runs_{study.study_id}",
        )
        st.markdown(
            "**For SAES**: the results (one row per run and indicator) and the metrics (all "
            "minimized), the two files SAES reads (`-ds` and `-ms`)."
        )
        saes_columns = st.columns(2)
        saes_columns[0].download_button(
            "Results for SAES (CSV)",
            saes_results(runs).to_csv(index=False),
            file_name=f"{study.study_id}_saes_results.csv",
            mime="text/csv",
            key=f"validation_saes_results_{study.study_id}",
            on_click="ignore",
            icon=":material/download:",
        )
        saes_columns[1].download_button(
            "Metrics for SAES (CSV)",
            saes_metrics(runs).to_csv(index=False),
            file_name=f"{study.study_id}_saes_metrics.csv",
            mime="text/csv",
            key=f"validation_saes_metrics_{study.study_id}",
            on_click="ignore",
            icon=":material/download:",
        )
    with tabs[4]:
        _render_details(study.directory, study.manifest)


def _render_history() -> None:
    """Choose a past study and show its results."""
    studies = {study.study_id: study for study in list_studies(RUNS_DIRECTORY)}
    if not studies:
        return
    st.subheader("Results")
    chosen = st.selectbox(
        "Previous studies",
        list(studies),
        index=None,
        placeholder="Choose a study to see its results",
        format_func=lambda study_id: studies[study_id].label,
        key=HISTORY_KEY,
    )
    if chosen is not None:
        _render_results(studies[chosen])


st.title("Validation")

jar = require_evolver_jar()
warn_if_jar_older_than_catalogue(jar)

catalogue = registered_problems(str(jar))
if catalogue is None:
    st.info(
        "This page needs an Evolver that describes its problems (2.4 or later): it chooses "
        "the algorithms to compare from the encoding of the problems."
    )
    st.stop()

in_progress = find_run_in_progress(RUNS_DIRECTORY)
if in_progress is not None:
    _render_study_in_progress(in_progress)
_render_form(catalogue, jar, can_run=in_progress is None)
_render_history()
