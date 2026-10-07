"""Tutorial S2, "Solving a problem with a configurable algorithm": its steps.

The interactive counterpart of Evolver's tutorial E2 (docs/tutorials/base_level_algorithms.rst): the
same story (configure an algorithm, run it, read its results, change its configuration, run it on
another problem), through the Run algorithm page instead of through Java code.

Every step that runs something has a *prepared run*: a request the tutorial can run right there
("Run it here"), with the values the text announces, and open in Run algorithm ("Open in Run
algorithm"), whose form it fills, so that the reader repeats the step by hand. The runs are fixed by
a seed, so they give the same values every time; those in the text were measured with Evolver 2.2.
"""

import datetime as dt
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

from evolver_studio import evolver_client
from evolver_studio.app_state import registered_problem_names
from evolver_studio.catalogue import BASE_ALGORITHMS
from evolver_studio.evolver_client import EVOLVER_VERSION, start_solve, write_pid_file
from evolver_studio.resource_files import default_configuration_text
from evolver_studio.runs import SOLVE_RUNS_DIRECTORY_NAME, RunPhase, run_phase
from evolver_studio.solve_figures import build_front_figure
from evolver_studio.solve_form import CONFIGURATION_VERSION_KEY, form_state_from_request
from evolver_studio.solve_request import SolveRequest, solve_request_to_yaml
from evolver_studio.solve_results import (
    filter_by_objectives,
    indicator_summary,
    read_front,
    read_indicators,
    read_run_fronts,
    read_solutions,
    variable_columns,
    zip_fronts,
)
from evolver_studio.tutorial_navigation import open_tutorial
from evolver_studio.tutorials import TutorialStep

E2_TUTORIAL_URL = (
    "https://github.com/jMetal/Evolver/blob/develop/docs/tutorials/base_level_algorithms.rst"
)
CLI_SOLVING_URL = "https://github.com/jMetal/Evolver/blob/develop/docs/utilities/cli_tools.rst"
KEY_PREFIX = "tutorial_solving"
TIMINGS_MEASURED_ON = (
    "Apple M5 Pro (18 cores), 64 GB of RAM, macOS 26.6.2, Java 21.0.12 (Oracle JDK)"
)
WEIGHT_VECTORS_DIRECTORY = "resources/weightVectors"
RUN_TIMEOUT_SECONDS = 180
# The values the text announces are written with four decimals.
EXPECTED_TOLERANCE = 6e-5

# The configuration of E2's step 3: a standard NSGA-II with little left of it.
OTHER_CONFIGURATION = (
    "--algorithmResult externalArchive --populationSizeWithArchive 20 "
    "--archiveType crowdingDistanceArchive --createInitialSolutions latinHypercubeSampling "
    "--offspringPopulationSize 10 --variation crossoverAndMutationVariation --crossover blxAlpha "
    "--crossoverProbability 0.9 --crossoverRepairStrategy bounds --blxAlphaCrossoverAlpha 0.5 "
    "--mutation uniform --mutationProbabilityFactor 1.0 --mutationRepairStrategy bounds "
    "--uniformMutationPerturbation 0.5 --selection tournament --selectionTournamentSize 2"
)


@dataclass(slots=True, frozen=True)
class PreparedRun:
    """A run the tutorial can launch, and the values it announces for it.

    Attributes:
        key: Its name in the tutorial, which keys the widgets and the run it remembers.
        algorithm: The algorithm's name in the catalogue (e.g. "NSGA-II").
        default_file: The default configuration it uses, a file of Evolver's defaultConfigurations/;
            None when `configuration` is given.
        configuration: The configuration, "--parameter value ...", when it is not a default one.
        problem: The problem's name.
        reference_front: Its reference front file.
        population_size: The population size.
        max_evaluations: The evaluations of each run.
        runs: The number of independent runs.
        expected: The indicator values the text announces (the first run's, or the mean of the runs
            when there are several), by short name.
    """

    key: str
    algorithm: str
    default_file: str | None
    configuration: str | None
    problem: str
    reference_front: str
    population_size: int
    max_evaluations: int
    runs: int
    expected: dict[str, float]


DEFAULT_ON_ZDT1 = PreparedRun(
    "default",
    "NSGA-II",
    "NSGAIIDoubleDefault.txt",
    None,
    "ZDT1",
    "resources/referenceFronts/ZDT1.csv",
    100,
    25000,
    1,
    {"EP": 0.0103, "NHV": 0.0098},
)
OTHER_ON_ZDT1 = PreparedRun(
    "other",
    "NSGA-II",
    None,
    OTHER_CONFIGURATION,
    "ZDT1",
    "resources/referenceFronts/ZDT1.csv",
    100,
    25000,
    1,
    {"EP": 0.0053, "NHV": 0.0062},
)
DEFAULT_ON_ZDT2_FIVE_RUNS = PreparedRun(
    "runs",
    "NSGA-II",
    "NSGAIIDoubleDefault.txt",
    None,
    "ZDT2",
    "resources/referenceFronts/ZDT2.csv",
    100,
    25000,
    5,
    {"EP": 0.0133, "NHV": 0.0197},
)
MOEAD_ON_DTLZ2 = PreparedRun(
    "moead",
    "MOEA/D",
    "MOEADDoubleDefault.txt",
    None,
    "DTLZ2",
    "resources/referenceFronts/DTLZ2.3D.csv",
    91,
    25000,
    1,
    {"EP": 0.0767, "NHV": 0.1220},
)
RVEA_VARIANTS = {
    "RVEA": PreparedRun(
        "rvea",
        "RVEA",
        "RVEADoubleDefault.txt",
        None,
        "DTLZ2",
        "resources/referenceFronts/DTLZ2.3D.csv",
        91,
        25000,
        1,
        {"EP": 0.0767, "NHV": 0.1231},
    ),
    "RVEA*": PreparedRun(
        "rveastar",
        "RVEA",
        "RVEAStarDoubleDefault.txt",
        None,
        "DTLZ2",
        "resources/referenceFronts/DTLZ2.3D.csv",
        91,
        25000,
        1,
        {"EP": 0.0914, "NHV": 0.1281},
    ),
    "iRVEA": PreparedRun(
        "irvea",
        "RVEA",
        "IRVEADoubleDefault.txt",
        None,
        "DTLZ2",
        "resources/referenceFronts/DTLZ2.3D.csv",
        91,
        25000,
        1,
        {"EP": 0.0926, "NHV": 0.1319},
    ),
}


def prepared_request(prepared: PreparedRun, jar: Path, output_directory: str) -> SolveRequest:
    """Build the solve request of a prepared run.

    Args:
        prepared: The run.
        jar: Path to Evolver's jar, to read the default configuration from.
        output_directory: Where the results are written.

    Returns:
        The request: seed 1, and the indicators EP and NHV.
    """
    algorithm = next(a for a in BASE_ALGORITHMS if a.name == prepared.algorithm)
    configuration = prepared.configuration or default_configuration_text(jar, prepared.default_file)
    extra_config = (
        {"weightVectorFilesDirectory": WEIGHT_VECTORS_DIRECTORY}
        if algorithm.required_extra_config_keys
        else None
    )
    return SolveRequest(
        algorithm_name=algorithm.registry_name,
        encoding="Double",
        population_size=prepared.population_size,
        yaml_parameter_space_file=algorithm.encodings["Double"],
        extra_config=extra_config,
        configuration=configuration.strip().splitlines()[0],
        problem=prepared.problem,
        reference_front_file_name=prepared.reference_front,
        max_evaluations=prepared.max_evaluations,
        number_of_independent_runs=prepared.runs,
        seed=1,
        indicator_names=["Epsilon", "NormalizedHypervolume"],
        # The tutorial's runs are short and its steps wait for each one to end: no live progress
        status_frequency=None,
        front_frequency=None,
        write_population=False,
        output_directory=output_directory,
    )


def run_prepared(prepared: PreparedRun, jar: Path) -> Path | None:
    """Run a prepared run now, and wait for it.

    The run is kept under solve-runs/, like any run of the Run algorithm page, so it appears in its
    list of previous runs.

    Args:
        prepared: The run.
        jar: Path to Evolver's jar.

    Returns:
        The run's directory, or None if it did not finish.
    """
    runs_directory = evolver_client.WORKING_DIRECTORY / SOLVE_RUNS_DIRECTORY_NAME
    moment = dt.datetime.now()
    while (runs_directory / moment.strftime("%Y%m%d-%H%M%S")).exists():
        moment += dt.timedelta(seconds=1)
    run_id = moment.strftime("%Y%m%d-%H%M%S")
    run_dir = runs_directory / run_id
    run_dir.mkdir(parents=True)
    request = prepared_request(prepared, jar, f"{SOLVE_RUNS_DIRECTORY_NAME}/{run_id}/output")
    (run_dir / "request.yaml").write_text(solve_request_to_yaml(request))
    process = start_solve(
        evolver_client.WORKING_DIRECTORY,
        jar,
        run_dir / "request.yaml",
        run_dir / "status.yaml",
        run_dir / "runner.log",
    )
    write_pid_file(run_dir / "pid.txt", process.pid)
    try:
        process.wait(timeout=RUN_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        return None
    deadline = time.monotonic() + 5
    while run_phase(run_dir) in (RunPhase.STARTING, RunPhase.RUNNING):
        if time.monotonic() > deadline:
            return None
        time.sleep(0.05)
    return run_dir if run_phase(run_dir) is RunPhase.FINISHED else None


def _output_directory(run_dir: Path) -> Path:
    """Where the results of a prepared run are: `run_prepared` writes them under its directory."""
    return run_dir / "output"


def _remembered_run(prepared: PreparedRun) -> Path | None:
    """The run the reader has already launched for a prepared run, if it still exists."""
    run_id = st.session_state.get(f"{KEY_PREFIX}_run_{prepared.key}")
    if run_id is None:
        return None
    run_dir = evolver_client.WORKING_DIRECTORY / SOLVE_RUNS_DIRECTORY_NAME / run_id
    return run_dir if run_phase(run_dir) is RunPhase.FINISHED else None


def _render_prepared_run(prepared: PreparedRun, jar: Path, show_front: bool = True) -> Path | None:
    """Offer a prepared run, run it on request, and show what it gave.

    Args:
        prepared: The run.
        jar: Path to Evolver's jar.
        show_front: Whether to plot the front the run found.

    Returns:
        The run's directory if it has been run, else None.
    """
    request = prepared_request(prepared, jar, "solve-runs/<run>/output")
    with st.container(border=True):
        st.markdown(
            f"**{prepared.algorithm}** on **{prepared.problem}**: population "
            f"{prepared.population_size}, {prepared.max_evaluations} evaluations, "
            f"{prepared.runs} run{'s' if prepared.runs > 1 else ''}, seed 1."
        )
        columns = st.columns(2)
        if columns[0].button("Run it here", key=f"{KEY_PREFIX}_button_{prepared.key}"):
            with st.spinner("Running…"):
                run_dir = run_prepared(prepared, jar)
            if run_dir is None:
                st.error("The run did not finish: see the Run algorithm page for its log.")
            else:
                st.session_state[f"{KEY_PREFIX}_run_{prepared.key}"] = run_dir.name
        if columns[1].button(
            "Open in Run algorithm",
            key=f"{KEY_PREFIX}_open_{prepared.key}",
            help="Fills the form of Run algorithm with this run, to repeat it by hand.",
        ):
            _open_in_run_algorithm(request, jar)
    run_dir = _remembered_run(prepared)
    if run_dir is not None:
        _render_outcome(prepared, run_dir, show_front)
    return run_dir


def _open_in_run_algorithm(request: SolveRequest, jar: Path) -> None:
    """Fill the Run algorithm form with a request and go to that page."""
    state = form_state_from_request(
        yaml.safe_load(solve_request_to_yaml(request)),
        BASE_ALGORITHMS,
        registered_problem_names(str(jar)) or [],
        st.session_state.get(CONFIGURATION_VERSION_KEY, 0),
    )
    if state is not None:
        st.session_state.update(state)
    st.switch_page("pages/solve.py")


def _render_outcome(prepared: PreparedRun, run_dir: Path, show_front: bool) -> None:
    output = _output_directory(run_dir)
    if show_front:
        fronts = read_run_fronts(output)
        reference = read_front(evolver_client.WORKING_DIRECTORY / prepared.reference_front)
        st.plotly_chart(
            build_front_figure(fronts, reference),
            width="stretch",
            key=f"{KEY_PREFIX}_front_{prepared.key}",
        )
    indicators = read_indicators(output)
    st.dataframe(indicators, hide_index=True, width="stretch")
    if len(indicators) > 1:
        st.dataframe(indicator_summary(indicators), width="stretch")
    _check_expected(prepared, indicators)


def _check_expected(prepared: PreparedRun, indicators: pd.DataFrame) -> None:
    """Say whether the indicator values are the ones the text announces."""
    values = {"EP": indicators["EP"], "NHV": indicators["NHV"]}
    measured = {
        name: series.mean() if len(series) > 1 else series.iloc[0]
        for name, series in values.items()
    }
    announced = ", ".join(f"{name} = {value}" for name, value in prepared.expected.items())
    matches = all(
        abs(measured[name] - value) <= EXPECTED_TOLERANCE
        for name, value in prepared.expected.items()
    )
    what = "mean over the runs" if len(indicators) > 1 else "run 1"
    if matches:
        st.success(f"The values announced ({announced}, {what}) are the ones you got.")
    else:
        got = ", ".join(f"{name} = {value:.4f}" for name, value in measured.items())
        st.info(
            f"The text announces {announced} ({what}); you got {got}. A different Evolver version "
            "can give other values."
        )


def _render_introduction(jar: Path) -> None:
    st.markdown(
        "Solving a problem with Evolver means choosing four things and running them: a "
        "**problem**, an **algorithm** (with its encoding), a **configuration** of that "
        "algorithm, and a **budget** (the population, the evaluations, how many independent "
        "runs). What you get is an approximation of the problem's Pareto **front** and the "
        "**quality indicators** that measure it against the problem's reference front.\n\n"
        "That is the page **Run algorithm**, which has a section for each of those four things "
        "and a button to run. In this tutorial you will:\n\n"
        "- run NSGA-II with its default configuration on a continuous problem and read the "
        "front and the indicators it gives;\n"
        "- see what else a run leaves: the solutions with their variables, and the files;\n"
        "- run it again with a very different configuration, adjusting the parameters you "
        "learned to read in tutorial S1;\n"
        "- repeat a run several times, and try a problem with three objectives and other "
        "algorithms;\n"
        "- take a run out of the app, to run it from a terminal.\n\n"
        "It is the interactive companion of Evolver's tutorial "
        f"[E2. Base-level algorithms]({E2_TUTORIAL_URL}), which does the same with Java code. "
        "This is **solving**: the configuration is something *you choose*. In a **training** "
        "(the subject of the coming tutorial *Your first guided training*) it is something "
        "Evolver *searches for*."
    )
    st.markdown(
        "Every run of this tutorial is prepared: **Run it here** runs it and shows the result, "
        "and **Open in Run algorithm** fills the form of that page with it, so that you can do it "
        "by hand. Each run uses the seed 1, so you should get the values the text gives."
    )
    st.caption(
        f"The runs of this tutorial take a few seconds or less each. Timings measured on: "
        f"{TIMINGS_MEASURED_ON}."
    )
    st.page_link("pages/solve.py", label="Open Run algorithm", icon="▶️")


def _render_default_run(jar: Path) -> None:
    st.markdown(
        "The first run is the standard one: NSGA-II with the default configuration Evolver ships "
        "for it, on the problem ZDT1 (two objectives, 30 variables), with a population of 100 and "
        "25000 evaluations. Its configuration, in the section 3 of Run algorithm, is:"
    )
    st.code(
        default_configuration_text(jar, DEFAULT_ON_ZDT1.default_file).strip(),
        language=None,
        wrap_lines=True,
    )
    st.markdown(
        "It is a standard NSGA-II: **SBX** crossover (`crossoverProbability` 0.9), **polynomial** "
        "mutation, **binary tournament** selection (`selectionTournamentSize` 2) and the final "
        "population as the result (`algorithmResult population`). The population size and the "
        "budget are not part of the configuration: they are chosen in the section 4."
    )
    _render_prepared_run(DEFAULT_ON_ZDT1, jar)
    st.markdown(
        "The front found is drawn over the reference front of ZDT1, the grey line. The table has "
        "one row per run with two indicators, **both to be minimized**: **EP** (Epsilon), how far "
        "the front is from the reference front, and **NHV** (normalized hypervolume), the fraction "
        "of the reference front's hypervolume that the front fails to cover. They are the same "
        "two that a training uses as its objectives."
    )
    answer = st.radio(
        "Quick check: which crossover does this configuration use?",
        ["SBX", "blxAlpha", "PCX"],
        index=None,
        key=f"{KEY_PREFIX}_crossover_quiz",
    )
    if answer == "SBX":
        st.success("Right: `--crossover SBX`, with a distribution index of 20.")
    elif answer is not None:
        st.error("Not quite: read the `--crossover` parameter of the configuration above.")


def _render_results(jar: Path) -> None:
    run_dir = _remembered_run(DEFAULT_ON_ZDT1)
    if run_dir is None:
        st.info("Run the first step to see its results here.")
        if st.button("Run it now", key=f"{KEY_PREFIX}_button_results"):
            with st.spinner("Running…"):
                finished = run_prepared(DEFAULT_ON_ZDT1, jar)
            if finished is not None:
                st.session_state[f"{KEY_PREFIX}_run_{DEFAULT_ON_ZDT1.key}"] = finished.name
                st.rerun()
        return
    output = _output_directory(run_dir)
    st.markdown(
        "After a run, the results of Run algorithm have four tabs: **Front** (the fronts of the "
        "runs over the reference front), **Indicators** (the table you saw, and a summary when "
        "there are several runs), **Solutions** and **Details**. A run keeps its files under "
        f"`{SOLVE_RUNS_DIRECTORY_NAME}/<run>/`:"
    )
    for path in sorted(output.rglob("*")):
        if path.is_file():
            st.write(f"- `{path.relative_to(output)}` ({path.stat().st_size} bytes)")
    st.markdown(
        "`FUN.csv` holds the objective values of each solution and `VAR.csv` its decision "
        "variables, one solution per line; `INDICATORS.csv` the table of indicators; "
        "`METADATA.txt` the settings of the run, including the configuration and the seeds."
    )
    solutions = read_solutions(output, 1)
    lower, upper = float(solutions["f1"].min()), float(solutions["f1"].max())
    st.markdown(
        "The **Solutions** tab lists them. Try its filter here: restrict `f1` to a range and see "
        "which solutions remain."
    )
    chosen = st.slider(
        "f1", lower, upper, (lower, upper), format="%.3f", key=f"{KEY_PREFIX}_f1_range"
    )
    shown = filter_by_objectives(solutions, {"f1": chosen})
    st.caption(f"{len(shown)} of {len(solutions)} solutions.")
    st.dataframe(shown, width="stretch")
    answer = st.radio(
        "Quick check: how many decision variables does a solution of ZDT1 have?",
        ["2", "30", "100"],
        index=None,
        key=f"{KEY_PREFIX}_variables_quiz",
    )
    if answer == "30":
        st.success(f"Right: the columns x1 to x{len(variable_columns(solutions))}.")
    elif answer is not None:
        st.error("Not quite: count the `x` columns of the table.")
    st.download_button(
        "Download the fronts and indicators (zip)",
        zip_fronts(output),
        file_name="tutorial-s3-step-1.zip",
        mime="application/zip",
        key=f"{KEY_PREFIX}_zip",
    )


def _render_other_configuration(jar: Path) -> None:
    st.markdown(
        "Changing the algorithm only means changing its configuration. In the section 3 of Run "
        "algorithm, the parameters of NSGA-II are shown with the default configuration's values "
        "and can be adjusted within the parameter space. Open the first run in Run algorithm and "
        "make these changes, which leave little of the standard NSGA-II:\n\n"
        "| Parameter | Value |\n|---|---|\n"
        "| `algorithmResult` | `externalArchive` (then `populationSizeWithArchive` 20 and "
        "`archiveType` `crowdingDistanceArchive` appear) |\n"
        "| `createInitialSolutions` | `latinHypercubeSampling` |\n"
        "| `offspringPopulationSize` | 10 |\n"
        "| `crossover` | `blxAlpha` (`sbxDistributionIndex` disappears and "
        "`blxAlphaCrossoverAlpha` 0.5 appears) |\n"
        "| `mutation` | `uniform` (`uniformMutationPerturbation` 0.5) |\n\n"
        "Notice how the parameters appear and disappear (the *active* parameters of tutorial S1), "
        "the ✏️ that marks each changed parameter and the count of parameters changed. "
        "**Reset to the default configuration** undoes everything."
    )
    st.page_link("pages/solve.py", label="Open Run algorithm", icon="▶️")
    st.markdown("Or run the prepared one here:")
    other = _render_prepared_run(OTHER_ON_ZDT1, jar, show_front=False)
    standard = _remembered_run(DEFAULT_ON_ZDT1)
    if other is not None and standard is not None:
        rows = []
        for label, run_dir in (
            ("Standard configuration", standard),
            ("Other configuration", other),
        ):
            indicators = read_indicators(_output_directory(run_dir))
            rows.append(
                {"": label, "EP": indicators["EP"].iloc[0], "NHV": indicators["NHV"].iloc[0]}
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.markdown(
            "Both configurations reach the reference front, but the second one spreads its "
            "solutions much more evenly along it: that is the effect of its **crowding-distance "
            "archive**, which keeps the solutions that are farthest apart from each other. In this "
            "run it also gets better values of both indicators. One run is not enough to conclude "
            "that it is better, though; that needs several runs and a statistical test (the "
            "coming tutorials *Comparing configurations on a problem* and *Validating a "
            "configuration*), but it shows how much the configuration matters, and why finding "
            "good ones automatically, which is what meta-optimization does, is worthwhile."
        )
    elif standard is None:
        st.info("Run the first step too, to compare the two.")
    st.markdown(
        "The values of the first run are the same as those of Evolver's tutorial E2, and the "
        "second ones are different because there the random numbers continue from the first run."
    )


def _render_more_runs(jar: Path) -> None:
    st.markdown(
        "One run is one sample of what a configuration can do. Run algorithm repeats it: "
        "**Independent runs** in the section 4. Each run has its own seed, **the first seed plus "
        "its number minus one** (1, 2, 3…), so that every run can be reproduced. Here, the "
        "default NSGA-II on ZDT2, five times:"
    )
    _render_prepared_run(DEFAULT_ON_ZDT2_FIVE_RUNS, jar)
    st.markdown(
        "The table has a row per run, with its seed and its time, and below it a summary with the "
        "mean, the standard deviation, the minimum and the maximum over the runs. The values "
        "change from run to run: the spread says how much of a difference between two "
        "configurations may be luck. Comparing configurations with several runs and a test is the "
        "subject of the coming tutorials *Comparing configurations on a problem* and "
        "*Validating a configuration*."
    )


def _render_more_objectives(jar: Path) -> None:
    st.markdown(
        "A front with three objectives is a surface. DTLZ2 has three objectives in its default "
        "form, and Run algorithm draws its front in 3D over the reference front (rotate it). The "
        "**reference front** must have the problem's number of objectives: DTLZ2 has fronts for "
        "2, 3, 4, 6 and 8 objectives, and you choose the right one (`DTLZ2.3D.csv`).\n\n"
        "**MOEA/D** is a decomposition algorithm: it spreads its population over a set of weight "
        "vectors read from a file (`W<objectives>D_<population>.dat`, in `resources/"
        "weightVectors`). That is why its population size in Run algorithm is a **selector** "
        "with the sizes that have a file for these three objectives (here, 91):"
    )
    _render_prepared_run(MOEAD_ON_DTLZ2, jar)
    st.markdown(
        "**RVEA** reads the weight vectors too, and has three variants, which differ in its "
        "`replacement` parameter: RVEA, RVEA* and iRVEA. Run algorithm offers the default "
        "configuration of each one:"
    )
    variant = st.radio(
        "Variant", list(RVEA_VARIANTS), horizontal=True, key=f"{KEY_PREFIX}_rvea_variant"
    )
    _render_prepared_run(RVEA_VARIANTS[variant], jar)
    st.caption(
        "A single run of each, on one problem: the values say what these runs gave, not which "
        "algorithm is better."
    )


def _render_take_it_elsewhere(jar: Path) -> None:
    run_dir = _remembered_run(DEFAULT_ON_ZDT1)
    st.markdown(
        "A run is a request, a YAML file, and Evolver can run it from a terminal: that is how "
        "Run algorithm runs it. In its results, **Use this configuration for a new run** fills the "
        "form with a past run (its problem, algorithm, configuration, budget and seed), to vary a "
        "parameter and run it again, and **Download request.yaml** gives the file. The request "
        "of the first run is:"
    )
    request = prepared_request(DEFAULT_ON_ZDT1, jar, "results/solve/NSGA-II.ZDT1")
    st.code(solve_request_to_yaml(request), language="yaml")
    st.markdown(
        "Save it as `request.yaml`, and from the directory that holds `resources/`, run:\n\n"
        f"```bash\njava -cp Evolver-{EVOLVER_VERSION}-jar-with-dependencies.jar \\\n"
        "    org.uma.evolver.cli.solving.SolveRunnerMain request.yaml\n```\n\n"
        "It writes the same files you saw (`run-1/FUN.csv`, `INDICATORS.csv`, …) in "
        "`outputDirectory`, and, next to the request, a `status.yaml` and a `results.yaml`. The "
        "requests are described in the documentation of the "
        f"[command line tools]({CLI_SOLVING_URL})."
    )
    if run_dir is None:
        st.info("Run the first step to see its files here.")
        return
    st.caption(
        f"Your first run is in `{SOLVE_RUNS_DIRECTORY_NAME}/{run_dir.name}/`, with the request it "
        "was started with, `request.yaml`."
    )


def _render_next_steps(jar: Path) -> None:
    st.markdown(
        "You can now configure an algorithm, run it on a problem and read what it gives. "
        "Three things to try on your own in **Run algorithm**:\n\n"
        "- Run NSGA-II with its default configuration on **ZDT3** or **ZDT4**. Do the indicators "
        "get better or worse than on ZDT1? Why would ZDT4 be harder?\n"
        "- Run **RVEA** with its three variants on **DTLZ2** with 2 objectives (you need the "
        "reference front `DTLZ2.2D.csv`, and a population size that has a weight vector file for "
        "two objectives).\n"
        "- Start from the default configuration of NSGA-II and change **one** parameter at a "
        "time (the crossover probability, the mutation distribution index). Which one moves the "
        "indicators the most? Use five runs, and *Use this configuration for a new run* to vary "
        "a past run.\n\n"
        "**What's next.** To find configurations like these automatically, the coming tutorial "
        "**Your first guided training** launches a first training. Evolver's tutorial "
        f"[E2]({E2_TUTORIAL_URL}) does what this one did with Java code, including a binary "
        "problem; running your own problem and comparing configurations rigorously come in "
        "later tutorials."
    )
    st.page_link("pages/solve.py", label="Open Run algorithm", icon="▶️")
    st.button(
        "All the tutorials",
        icon="🎓",
        on_click=open_tutorial,
        args=(None,),
        key=f"{KEY_PREFIX}_all_tutorials",
    )


STEPS: tuple[TutorialStep, ...] = (
    TutorialStep("What does it mean to solve a problem?", _render_introduction),
    TutorialStep("Step 1: the default NSGA-II on ZDT1", _render_default_run),
    TutorialStep("Step 2: the results", _render_results),
    TutorialStep("Step 3: a different configuration", _render_other_configuration),
    TutorialStep("Step 4: more runs", _render_more_runs),
    TutorialStep("Step 5: more objectives, other algorithms", _render_more_objectives),
    TutorialStep("Step 6: take it elsewhere", _render_take_it_elsewhere),
    TutorialStep("Try it yourself", _render_next_steps),
)
