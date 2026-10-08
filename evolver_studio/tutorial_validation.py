"""Tutorial S5, "Validating a configuration": its steps.

The interactive counterpart of Evolver's tutorial E8
(docs/tutorials/validating_a_configuration.rst): a configuration found by a training is run again,
many times, next to other algorithms, and the results are compared with a statistical test and an
effect size, through the Validation page instead of through Java code and scripts.

The tutorial runs one study, with the configuration that Evolver's tutorial E8 found for NSGA-II on
the nine bi-objective WFG problems (bundled in Evolver's jar) as the pivot. The study is kept under
validation-runs/ like any other, so the Validation page shows it too. Its runs are fixed by the
seeds, so they give the same values every time; and what the text says about the results is read
from them, not written down, so that it holds for the Evolver in use.
"""

import os
import shutil
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from evolver_studio import evolver_client, running_badge
from evolver_studio.app_state import registered_problems
from evolver_studio.progress import study_running_label
from evolver_studio.resource_files import (
    default_configuration_text,
    jar_evolver_version,
    tuned_configuration_text,
)
from evolver_studio.runs import RunPhase, run_phase
from evolver_studio.tutorial_navigation import open_tutorial
from evolver_studio.tutorials import TutorialStep
from evolver_studio.validation import (
    Contender,
    StudyProblem,
    ValidationStudy,
    collect_runs,
    default_contenders,
    study_is_written,
    write_study,
)
from evolver_studio.validation_form import form_state_from_study
from evolver_studio.validation_runner import VALIDATION_RUNS_DIRECTORY_NAME, start_study
from evolver_studio.validation_stats import (
    NO_DIFFERENCE,
    PIVOT_BETTER,
    PIVOT_WORSE,
    best_contenders,
    compare_with_pivot,
    indicator_names,
    interquartile_ranges,
    medians,
)

E8_TUTORIAL_URL = (
    "https://github.com/jMetal/Evolver/blob/develop/docs/tutorials/validating_a_configuration.rst"
)
KEY_PREFIX = "tutorial_validation"
STUDY_ID = "tutorial-validation"
VERSION_MARKER = "evolver_version.txt"
TIMINGS_MEASURED_ON = (
    "Apple M5 Pro (18 cores), 64 GB of RAM, macOS 26.6.2, Java 21.0.12 (Oracle JDK)"
)
TUNED_FILE = "NSGAIIWFG2D.txt"
GECCO_2019_FILE = "NSGAIIWFG2DGECCO2019.txt"
PIVOT_NAME = "NSGA-II (tuned)"
COMPARED_WITH = ("NSGA-II", "MOEA/D", "RVEA")
REFERENCE_FRONTS = "resources/referenceFronts/"
SEEN_PROBLEMS = ("WFG2", "WFG4")
POPULATION_SIZE = 100
MAX_EVALUATIONS = 25000
RUNS = 15
SEED = 1
MAX_PARALLEL_JOBS = 4
POLL_EVERY_SECONDS = 1
RUNS_SHOWN = tuple(range(5, RUNS + 1))


def tutorial_study(jar: Path) -> ValidationStudy:
    """The study of the tutorial.

    Args:
        jar: Path to Evolver's jar, which holds the tuned and the default configurations.

    Returns:
        The tuned NSGA-II against the default NSGA-II, MOEA/D and RVEA, on two WFG problems it
        was tuned for and two others (ZDT1, and DTLZ2 with two objectives), with the budget of
        Evolver's tutorial E8.
    """
    tuned = tuned_configuration_text(jar, TUNED_FILE).strip().splitlines()[0]
    defaults = {
        contender.name: contender
        for contender in default_contenders(
            "Double", lambda file_name: default_configuration_text(jar, file_name)
        )
    }
    return ValidationStudy(
        encoding="Double",
        problems=(
            StudyProblem("WFG2", (), f"{REFERENCE_FRONTS}WFG2.2D.csv"),
            StudyProblem("WFG4", (), f"{REFERENCE_FRONTS}WFG4.2D.csv"),
            StudyProblem("ZDT1", (), f"{REFERENCE_FRONTS}ZDT1.csv"),
            StudyProblem("DTLZ2", (12, 2), f"{REFERENCE_FRONTS}DTLZ2.2D.csv"),
        ),
        contenders=(
            Contender(PIVOT_NAME, "NSGA-II", tuned),
            *(defaults[name] for name in COMPARED_WITH),
        ),
        pivot=PIVOT_NAME,
        population_size=POPULATION_SIZE,
        max_evaluations=MAX_EVALUATIONS,
        runs=RUNS,
        seed=SEED,
    )


def _needs_problem_catalogue(render: Callable[[Path], None]) -> Callable[[Path], None]:
    """Make a step say so when the Evolver in use cannot run the study.

    The study gives DTLZ2 its arguments, which an Evolver that does not describe its problems
    (2.3 and older) does not take, like the Validation page.
    """

    def step(jar: Path) -> None:
        if registered_problems(str(jar)) is None:
            st.info(
                "This step needs an Evolver that describes its problems (2.4 or later): the "
                "study gives DTLZ2 its number of objectives and variables."
            )
            return
        render(jar)

    step.__name__ = render.__name__
    return step


def _study_directory() -> Path:
    return evolver_client.WORKING_DIRECTORY / VALIDATION_RUNS_DIRECTORY_NAME / STUDY_ID


def _is_current(jar: Path) -> bool:
    """Whether the study kept on disk is the tutorial's, run with the Evolver in use."""
    directory = _study_directory()
    marker = directory / VERSION_MARKER
    return (
        study_is_written(tutorial_study(jar), directory, evolver_client.WORKING_DIRECTORY)
        and marker.is_file()
        and marker.read_text() == str(jar_evolver_version(jar))
    )


def _start(jar: Path) -> None:
    """Write the study, replacing any earlier one of the tutorial, and run it."""
    directory = _study_directory()
    shutil.rmtree(directory, ignore_errors=True)
    write_study(tutorial_study(jar), directory, evolver_client.WORKING_DIRECTORY)
    (directory / VERSION_MARKER).write_text(str(jar_evolver_version(jar)))
    processes = min(os.cpu_count() or 1, MAX_PARALLEL_JOBS)
    start_study(directory, jar, processes, evolver_client.WORKING_DIRECTORY)


def _phase(jar: Path) -> RunPhase | None:
    """Where the tutorial's study is, or None when there is no current one to talk about."""
    return run_phase(_study_directory()) if _is_current(jar) else None


def _render_run_controls(jar: Path) -> bool:
    """Offer to run the study, show its progress, and say when it is ready.

    Args:
        jar: Path to Evolver's jar.

    Returns:
        Whether the results of the study are ready.
    """
    phase = _phase(jar)
    if phase in (RunPhase.STARTING, RunPhase.RUNNING):
        _render_progress()
        return False
    if phase is RunPhase.FINISHED:
        st.success(
            f"The study is ready. It is kept as **{STUDY_ID}** in the Validation page, which "
            "shows its results too."
        )
        if st.button("Run it again", key=f"{KEY_PREFIX}_again"):
            _start(jar)
            st.rerun()
        return True
    if phase in (RunPhase.FAILED, RunPhase.LOST, RunPhase.CANCELLED):
        st.error("The last attempt did not finish: see Validation for the runner's log.")
    if st.button("Run the study", type="primary", key=f"{KEY_PREFIX}_run"):
        _start(jar)
        st.rerun()
    return False


def _render_progress() -> None:
    """Show that the study is in progress, and move on when it ends."""
    running_badge.inject_style()

    @st.fragment(run_every=POLL_EVERY_SECONDS, key=f"{KEY_PREFIX}_poll")
    def _poll() -> None:
        status = evolver_client.read_status(_study_directory() / "status.yaml")
        running_badge.render(study_running_label(status))
        if run_phase(_study_directory()) not in (RunPhase.STARTING, RunPhase.RUNNING):
            st.rerun()

    _poll()


def _runs_or_ask_to_run(jar: Path) -> pd.DataFrame | None:
    """The runs of the study, or the controls to run it when they are not there yet."""
    if _phase(jar) is RunPhase.FINISHED:
        runs = collect_runs(_study_directory(), evolver_client.WORKING_DIRECTORY)
        if not runs.empty:
            return runs
    st.info("The study has not been run yet: run it here to see its results.")
    _render_run_controls(jar)
    return None


def _choose_indicator(runs: pd.DataFrame, key: str, with_doubtful_case: bool = False) -> str:
    """Choose the indicator a step looks at.

    Args:
        runs: The runs of the study.
        key: The widget's key, within the tutorial.
        with_doubtful_case: Whether to start on the first indicator that has a comparison the
            test does not settle, which is what the step is about, instead of the first one.

    Returns:
        The indicator.
    """
    indicators = indicator_names(runs)
    index = 0
    if with_doubtful_case:
        index = next(
            (
                i
                for i, indicator in enumerate(indicators)
                if _doubtful(compare_with_pivot(runs, indicator, PIVOT_NAME)) is not None
            ),
            0,
        )
    return st.radio(
        "Indicator", indicators, index=index, horizontal=True, key=f"{KEY_PREFIX}_{key}"
    )


def _render_introduction(jar: Path) -> None:
    st.markdown(
        "A training ends with a configuration that was good on the training problems, in the "
        "runs of the training. Whether it is good, and where, is the question of a **validation "
        "study**. The values a training reports cannot answer it:\n\n"
        "- **They are biased.** The configuration was chosen *because* its values were the best "
        "of thousands, so part of its advantage is luck.\n"
        "- **They use the training budget**, smaller than the usual one.\n"
        "- **They only cover the training problems.**\n\n"
        "A validation runs the configuration again, with new seeds and the usual budget, on the "
        "training problems and on others, **many times**, next to other algorithms; and judges "
        "the differences with statistical tests, because one run says little about a stochastic "
        "algorithm.\n\n"
        "In this tutorial you will run a small validation study and read it: medians, a "
        "Wilcoxon test, an effect size, and what the number of runs has to do with it. It is the "
        f"interactive companion of Evolver's tutorial [E8. Validating a configuration]"
        f"({E8_TUTORIAL_URL}), which runs a bigger study with Java code and scripts and adds "
        "the tests that this page does not have yet."
    )
    st.markdown(
        "The study is made in **Validation**, the page where you will make your own. Here every "
        "step reads the results of one study that the tutorial runs for you."
    )


def _design_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            (
                "Tuned configuration (the pivot)",
                f"{TUNED_FILE}, in Evolver's jar: NSGA-II "
                "tuned for the nine bi-objective WFG problems (tutorial E8)",
            ),
            ("Compared with", "the default configurations of NSGA-II, MOEA/D and RVEA"),
            ("Problems it was tuned for", ", ".join(SEEN_PROBLEMS)),
            ("Problems it never saw", "ZDT1, and DTLZ2 with two objectives (arguments 12, 2)"),
            (
                "Budget of every algorithm",
                f"population {POPULATION_SIZE}, {MAX_EVALUATIONS:,} "
                f"evaluations per run, {RUNS} independent runs (seeds {SEED} to {SEED + RUNS - 1})",
            ),
            ("Indicators", "EP and NHV, both to be minimized"),
        ],
        columns=["Choice", "In this study"],
    )


@_needs_problem_catalogue
def _render_study(jar: Path) -> None:
    study = tutorial_study(jar)
    st.markdown(
        "A study answers a question, and every choice follows from it. Ours: *is the tuned "
        "NSGA-II better than the standard NSGA-II and than two algorithms of other families, on "
        "the problems it was tuned for and on others?*"
    )
    st.table(_design_table().set_index("Choice"))
    st.markdown(
        "The same population size, evaluation budget and seeds for everyone: run *i* of every "
        "algorithm uses the seed *i*, so that the comparison is fair. Four algorithms on four "
        f"problems are **{len(study.contenders) * len(study.problems)} jobs** of {RUNS} runs, "
        f"{len(study.contenders) * len(study.problems) * RUNS} runs in all. It takes about half "
        f"a minute on {TIMINGS_MEASURED_ON}, and more on a slower machine; it goes on in the "
        "background if you leave this step."
    )
    with st.expander("The tuned configuration"):
        st.code(study.contenders[0].configuration, language=None, wrap_lines=True)
    _render_run_controls(jar)
    if st.button(
        "Open in Validation",
        key=f"{KEY_PREFIX}_open",
        help="Fills the form of Validation with this study, to make it by hand.",
    ):
        st.session_state.update(form_state_from_study(study))
        st.switch_page("pages/validation.py")


@_needs_problem_catalogue
def _render_medians(jar: Path) -> None:
    runs = _runs_or_ask_to_run(jar)
    if runs is None:
        return
    indicator = _choose_indicator(runs, "medians_indicator")
    table = medians(runs, indicator)
    st.markdown(
        f"The **median of {indicator}** over the {RUNS} runs of each algorithm on each problem, "
        "lower is better; the best of each row is highlighted. The summary is the median, not "
        "the mean: the values of a metaheuristic are rarely normally distributed, and a few runs "
        "stuck in a poor front move the mean far from the typical run, but not the median."
    )
    st.dataframe(
        table.style.highlight_min(axis=1).format("{:.4g}"), width="stretch", height="content"
    )
    if (table < 0).any().any():
        st.caption(
            "A negative NHV means that the front covers a little more hypervolume than the "
            "reference front does."
        )
    with st.expander("How spread out are the runs? Interquartile range and boxplots"):
        st.markdown(
            "The **interquartile range** is the width of the interval that holds the central half "
            "of the runs: a small one means that the algorithm is consistent. A boxplot draws it: "
            "the box goes from the first to the third quartile, the line is the median."
        )
        st.dataframe(
            interquartile_ranges(runs, indicator).style.format("{:.4g}"),
            width="stretch",
            height="content",
        )
        st.plotly_chart(_boxplot(runs, indicator), width="stretch", key=f"{KEY_PREFIX}_boxplot")
    _render_best_quiz(table, indicator)


def _boxplot(runs: pd.DataFrame, indicator: str):
    figure = px.box(
        runs,
        x="contender",
        y=indicator,
        color="contender",
        facet_col="problem",
        facet_col_wrap=2,
        labels={"contender": ""},
    )
    figure.update_yaxes(matches=None, showticklabels=True)
    figure.update_xaxes(showticklabels=False)
    figure.update_layout(height=520)
    return figure


def _render_best_quiz(table: pd.DataFrame, indicator: str) -> None:
    problem = "ZDT1"
    answer = st.radio(
        f"Quick check: which algorithm has the lowest median {indicator} on {problem}?",
        list(table.columns),
        index=None,
        key=f"{KEY_PREFIX}_best_quiz_{indicator}",
    )
    best = best_contenders(table)[problem]
    if answer == best:
        st.success(f"Right: {best}, with {table.loc[problem, best]:.4g}.")
    elif answer is not None:
        st.error("Not quite: read the row of ZDT1 in the table above.")


def _doubtful(comparison: pd.DataFrame) -> pd.Series | None:
    """The comparison where the pivot is least clearly better: the one with the highest p-value."""
    not_better = comparison[comparison["verdict"] != PIVOT_BETTER]
    if not_better.empty:
        return None
    return not_better.sort_values("p_value", ascending=False).iloc[0]


@_needs_problem_catalogue
def _render_significance(jar: Path) -> None:
    runs = _runs_or_ask_to_run(jar)
    if runs is None:
        return
    indicator = _choose_indicator(runs, "significance_indicator", with_doubtful_case=True)
    comparison = compare_with_pivot(runs, indicator, PIVOT_NAME)
    st.markdown(
        f"A lower median does not make a better algorithm: with {RUNS} runs it can be luck. "
        f"Two numbers say more, for the tuned NSGA-II against each other algorithm on each "
        "problem:\n\n"
        "- the **p-value** of a **Wilcoxon rank-sum test**: the probability of seeing a "
        "difference at least this large if the two algorithms were equally good. Below 0.05 "
        "the difference is called *significant*;\n"
        "- the **A12 effect size** of Vargha and Delaney: the probability that a run of the "
        "tuned NSGA-II is better than a run of the other algorithm. 0.5 is no difference, 1 is "
        "always better; the effect is *small*, *medium* or *large* from 0.56, 0.64 and 0.71. A "
        "significant difference can be tiny, and a large one can fail to be significant: the "
        "test and the effect size answer different questions."
    )
    detail = comparison.rename(
        columns={
            "problem": "Problem",
            "contender": "Algorithm",
            "pivot_median": "Tuned median",
            "median": "Median",
            "p_value": "p-value",
            "a12": "A12",
            "magnitude": "Effect",
            "verdict": "Verdict",
        }
    )
    st.dataframe(
        detail.style.format(
            {"Tuned median": "{:.4g}", "Median": "{:.4g}", "p-value": "{:.4f}", "A12": "{:.2f}"}
        ),
        hide_index=True,
        width="stretch",
        height="content",
    )
    _render_doubtful_case(comparison, indicator)


def _render_doubtful_case(comparison: pd.DataFrame, indicator: str) -> None:
    case = _doubtful(comparison)
    if case is None:
        st.success(
            f"On {indicator}, the tuned NSGA-II is significantly better in every comparison of "
            "this study. Look at the other indicator: it may not be."
        )
        return
    lower = "lower" if case["pivot_median"] < case["median"] else "higher"
    st.markdown(
        f"**Look at {case['problem']}, against {case['contender']}.** The tuned median of "
        f"{indicator} is {case['pivot_median']:.4g} and {case['contender']}'s "
        f"{case['median']:.4g}: the tuned one is {lower}. The test gives "
        f"p = {case['p_value']:.3f} and the A12 is {case['a12']:.2f} ({case['magnitude']})."
    )
    answer = st.radio(
        f"Quick check: can we say that the tuned NSGA-II is better than {case['contender']} on "
        f"{case['problem']}, on {indicator}?",
        [
            "Yes: its median is lower",
            "Not with these runs: the difference is not significant",
            "No: it is worse",
        ],
        index=None,
        key=f"{KEY_PREFIX}_doubtful_quiz_{indicator}",
    )
    if case["verdict"] == NO_DIFFERENCE:
        if answer == "Not with these runs: the difference is not significant":
            st.success("Right. The next step shows what more runs could change.")
        elif answer is not None:
            st.error("Not quite: look at the p-value and compare it with 0.05.")
    elif answer is not None:
        st.info(f"Here the verdict is: {case['verdict']}.")


def _runs_subset(runs: pd.DataFrame, count: int) -> pd.DataFrame:
    """The first `count` runs of every algorithm on every problem (seeds 1 to `count`)."""
    return runs[runs["Run"] <= count]


@_needs_problem_catalogue
def _render_number_of_runs(jar: Path) -> None:
    runs = _runs_or_ask_to_run(jar)
    if runs is None:
        return
    st.markdown(
        "The less luck weighs, the more runs a test has to work with. Take the comparison of "
        "the last step that is least clear and ask: what would the test have said with fewer "
        "runs? Run *i* always uses the seed *i*, so the first *n* runs are exactly what a "
        "study of *n* runs would have run."
    )
    indicator = _choose_indicator(runs, "runs_indicator", with_doubtful_case=True)
    comparison = compare_with_pivot(runs, indicator, PIVOT_NAME)
    case = _doubtful(comparison)
    if case is None:
        st.info(f"Every comparison on {indicator} is significant: try the other indicator.")
        return
    rows = []
    for count in RUNS_SHOWN:
        subset = compare_with_pivot(_runs_subset(runs, count), indicator, PIVOT_NAME)
        row = subset[
            (subset["problem"] == case["problem"]) & (subset["contender"] == case["contender"])
        ].iloc[0]
        rows.append(
            {
                "Runs": count,
                "p-value": row["p_value"],
                "A12": row["a12"],
                "Effect": row["magnitude"],
                "Verdict": row["verdict"],
            }
        )
    st.markdown(
        f"The tuned NSGA-II against {case['contender']} on {case['problem']}, on {indicator}, "
        "with the first *n* runs:"
    )
    st.dataframe(
        pd.DataFrame(rows).style.format({"p-value": "{:.4f}", "A12": "{:.2f}"}),
        hide_index=True,
        width="stretch",
        height="content",
    )
    table = pd.DataFrame(rows)
    trend = (
        "It falls as runs are added, but every new run can favour either algorithm."
        if table["p-value"].is_monotonic_decreasing
        else "It does not fall steadily: every new run can favour either algorithm, and a "
        "verdict can change with one of them."
    )
    st.markdown(
        f"With {RUNS_SHOWN[0]} to {RUNS_SHOWN[-1]} runs the p-value goes from "
        f"{table['p-value'].min():.3f} to {table['p-value'].max():.3f} and the A12 from "
        f"{table['A12'].min():.2f} to {table['A12'].max():.2f}. {trend} What "
        "more runs give is **power**, the chance of detecting a difference when there is one, "
        "and a verdict that depends less on a lucky seed; they cannot create a difference. "
        "*No significant difference* means that the study does not show one, not that the two "
        "algorithms are equal.\n\n"
        "That is why validations use 25 to 30 runs (Evolver's tutorial E8 uses 25), and why this "
        f"one, with {RUNS} to take half a minute, is a sketch. In Validation, raise "
        "**Independent runs** and run the study again to see what changes."
    )


def _count(comparison: pd.DataFrame, problems: tuple[str, ...] | list[str]) -> tuple[int, int]:
    rows = comparison[comparison["problem"].isin(problems)]
    return int((rows["verdict"] == PIVOT_BETTER).sum()), len(rows)


def _count_verdict(
    comparison: pd.DataFrame, problems: tuple[str, ...] | list[str], verdict: str
) -> int:
    rows = comparison[comparison["problem"].isin(problems)]
    return int((rows["verdict"] == verdict).sum())


@_needs_problem_catalogue
def _render_seen_and_unseen(jar: Path) -> None:
    runs = _runs_or_ask_to_run(jar)
    if runs is None:
        return
    st.markdown(
        "The tuning only saw the WFG problems. A configuration that is good on them may be good "
        "only on them, which is why a study includes problems it never saw. Here, in how many "
        "of the comparisons (three algorithms on each problem) the tuned NSGA-II is significantly "
        "better:"
    )
    seen = list(SEEN_PROBLEMS)
    unseen = [p for p in dict.fromkeys(runs["problem"]) if p not in seen]
    rows = []
    totals = {"Tuned for": [0, 0], "Never seen": [0, 0]}
    for indicator in indicator_names(runs):
        comparison = compare_with_pivot(runs, indicator, PIVOT_NAME)
        for group, problems in (("Tuned for", seen), ("Never seen", unseen)):
            better, total = _count(comparison, problems)
            worse = _count_verdict(comparison, problems, PIVOT_WORSE)
            totals[group][0] += better
            totals[group][1] += total
            rows.append(
                {
                    "Indicator": indicator,
                    "Problems": f"{group}: {', '.join(problems)}",
                    "Significantly better": f"{better} of {total}",
                    "Significantly worse": f"{worse} of {total}",
                }
            )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height="content")
    st.markdown(
        f"Over both indicators, the tuned NSGA-II is significantly better in "
        f"{totals['Tuned for'][0]} of {totals['Tuned for'][1]} comparisons on the problems it "
        f"was tuned for, and in {totals['Never seen'][0]} of {totals['Never seen'][1]} on the "
        "ones it never saw.\n\n"
        "Whether the advantage carries over to problems of other families is not guaranteed: "
        "it is what the validation measures. A configuration tuned for a family of problems can "
        "lose to the default one on another, and then the honest conclusion is that it is a "
        "good configuration *for that family*."
    )


def _render_next_steps(jar: Path) -> None:
    st.markdown(
        "**What a validation does not tell.**\n\n"
        "- **The budget.** The conclusions hold for 25,000 evaluations; a longer or a shorter "
        "run can order the algorithms differently.\n"
        "- **The indicators.** EP measures convergence and NHV convergence and spread together; "
        "Spread or IGD+ may disagree.\n"
        "- **Many comparisons.** Twelve tests at the 5% level are expected to flag a difference "
        "now and then by chance. Evolver's tutorial E8 analyzes all the problems at once, with "
        "the Friedman test and Holm's procedure, and with critical difference plots; Validation "
        "does not have them yet.\n"
        "- **The problems.** Four of them are a small sample of what a configuration may meet.\n\n"
        "**Explore on your own**, in Validation:\n\n"
        "- Add **SMS-EMOA** or another algorithm to compare with (the ones with a default "
        "configuration for the encoding are offered), or other problems, such as the other WFG "
        "problems or the DTLZ ones.\n"
        "- Raise the **independent runs** to 30 and see which verdicts change.\n"
        "- Choose **Binary** or **Permutation** problems: the algorithms offered change with "
        "the encoding.\n"
        "- Paste another tuned configuration as the pivot, such as the one Evolver found in the "
        "smaller parameter space of the 2019 paper (Evolver's tutorial E5), to see whether it is "
        "better than the first one. Here it is:"
    )
    st.code(
        tuned_configuration_text(jar, GECCO_2019_FILE).strip().splitlines()[0],
        language=None,
        wrap_lines=True,
    )
    st.markdown(
        "The configurations of your own trainings are offered by Validation directly: choose "
        "*A training run* and one of the configurations of its final front."
    )
    st.page_link("pages/validation.py", label="Open Validation", icon="✅")
    st.button(
        "All the tutorials",
        icon="🎓",
        on_click=open_tutorial,
        args=(None,),
        key=f"{KEY_PREFIX}_all_tutorials",
    )


STEPS: tuple[TutorialStep, ...] = (
    TutorialStep("What does a validation answer?", _render_introduction),
    TutorialStep("Step 1: design and run the study", _render_study),
    TutorialStep("Step 2: medians", _render_medians),
    TutorialStep("Step 3: is the difference real?", _render_significance),
    TutorialStep("Step 4: how many runs?", _render_number_of_runs),
    TutorialStep("Step 5: tuned-for and unseen problems", _render_seen_and_unseen),
    TutorialStep("What it does not tell, and what to try", _render_next_steps),
)
