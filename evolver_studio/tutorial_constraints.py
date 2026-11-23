"""Tutorial "Problems with constraints": its steps.

The interactive counterpart of Evolver's tutorial E12
(docs/tutorials/problems_with_constraints.rst), on the same two problems, DASCMOP2 and DASCMOP5:
what a constraint-handling technique does (NSGA-II without one, with cdp and with push and pull
search), how a run is measured when it may find no feasible solution, and a training that
configures NSGA-II, its technique included, for DASCMOP5, which the reader launches in Training.

What the text says about the runs is computed from them, so it holds for whatever they give.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

from evolver_studio import evolver_client
from evolver_studio.analysis_view import monitor_state, render_summary
from evolver_studio.app_state import registered_problems
from evolver_studio.catalogue import BASE_ALGORITHMS, CONSTRAINT_HANDLING_PARAMETERS
from evolver_studio.constraint_handling import CONSTRAINT_HANDLING
from evolver_studio.monitor_view import render_feasibility, render_population
from evolver_studio.parameter_space import CategoricalParameter, parse_parameter_space
from evolver_studio.problem_browser import render_table
from evolver_studio.problem_catalogue import WITH_CONSTRAINTS
from evolver_studio.resource_files import default_configuration_text, parameter_space_text
from evolver_studio.solve_results import feasible_runs, read_indicators, read_run_violations
from evolver_studio.training_presets import open_training, training_state
from evolver_studio.training_runs import FinishedTraining, list_finished_trainings
from evolver_studio.tutorial_navigation import open_tutorial
from evolver_studio.tutorial_solving import (
    PreparedRun,
    output_directory,
    remembered_run,
    render_prepared_run,
)
from evolver_studio.tutorials import TutorialStep

E12_TUTORIAL_URL = (
    "https://github.com/jMetal/Evolver/blob/develop/docs/tutorials/problems_with_constraints.rst"
)
KEY_PREFIX = "tutorial_constraints"
ALGORITHM = "NSGA-II"
SIMPLE_PROBLEM = "DASCMOP2"
HARD_PROBLEM = "DASCMOP5"
POPULATION_SIZE = 100
EVALUATIONS = 20000
RUNS = 10
INDICATORS = ("ConstrainedMultiobjectiveIndicator", "ConstrainedEpsilon")
# The ICMOP of a run above this threshold is that of a run without feasible solutions.
THRESHOLD = 1.0
META_OPTIMIZER = "AsyncNSGA-II"
META_EVALUATIONS = 2000
META_POPULATION_SIZE = 50
TIMINGS_MEASURED_ON = "Apple M5 Pro (18 cores), 64 GB of RAM, macOS 26.6.2, Java 21"
PUSH_PULL = (
    "--constraintHandling pushPull --pushPullChangeThreshold 0.001 --pushPullWindowFraction 0.05 "
    "--pushPullDecayFraction 0.8 --pushPullExponent 2.0 --pushPullFeasibilityThreshold 0.95 "
    "--pushPullReductionRate 0.1"
)


@dataclass(slots=True, frozen=True)
class Outcome:
    """What some runs on a problem with constraints gave.

    Attributes:
        feasible: How many of them found feasible solutions.
        runs: How many they are.
        icmop: The median of their ICMOP.
    """

    feasible: int
    runs: int
    icmop: float


def prepared_run(
    key: str, problem: str, default_file: str | None, configuration: str | None = None
) -> PreparedRun:
    """A run of this tutorial: NSGA-II, 10 times, measured with the indicators for constraints.

    Args:
        key: Its name in the tutorial.
        problem: The problem.
        default_file: The default configuration it uses, or None when `configuration` is given.
        configuration: The configuration, when it is not a default one.
    """
    return PreparedRun(
        f"constraints_{key}",
        ALGORITHM,
        default_file,
        configuration,
        problem,
        f"resources/referenceFronts/{problem}.csv",
        POPULATION_SIZE,
        EVALUATIONS,
        RUNS,
        {},
        INDICATORS,
    )


def push_pull_configuration(jar: Path) -> str:
    """The default configuration of NSGA-II for constraints, with push and pull search."""
    default = default_configuration_text(jar, "NSGAIIDoubleConstrainedDefault.txt").strip()
    return default.replace("--constraintHandling cdp", PUSH_PULL)


def outcome(prepared: PreparedRun) -> Outcome | None:
    """What a run of this tutorial gave; None if it has not been run."""
    run_dir = remembered_run(prepared)
    if run_dir is None:
        return None
    output = output_directory(run_dir)
    indicators = read_indicators(output)
    violations = read_run_violations(output, indicators["Run"])
    return Outcome(feasible_runs(violations), len(indicators), float(indicators["ICMOP"].median()))


def training_form_state() -> dict[str, object]:
    """The values of the Training form for this tutorial's training, by widget key."""
    return training_state(
        algorithm=ALGORITHM,
        problems=(HARD_PROBLEM,),
        meta_algorithm=META_OPTIMIZER,
        meta_evaluations=META_EVALUATIONS,
        meta_population=META_POPULATION_SIZE,
        update_every=META_POPULATION_SIZE,
        population_size=POPULATION_SIZE,
        group=WITH_CONSTRAINTS,
    )


def latest_tutorial_training() -> FinishedTraining | None:
    """The most recent finished training of NSGA-II for this tutorial's problem, if any."""
    for training in list_finished_trainings(evolver_client.WORKING_DIRECTORY):
        if training.algorithm == ALGORITHM and training.problems == (HARD_PROBLEM,):
            return training
    return None


def _without(problem: str) -> PreparedRun:
    return prepared_run(f"without_{problem}", problem, "NSGAIIDoubleDefault.txt")


def _cdp(problem: str) -> PreparedRun:
    return prepared_run(f"cdp_{problem}", problem, "NSGAIIDoubleConstrainedDefault.txt")


def _push_pull(jar: Path) -> PreparedRun:
    return prepared_run("push_pull", HARD_PROBLEM, None, push_pull_configuration(jar))


def _render_comparison(prepared: dict[str, PreparedRun]) -> dict[str, Outcome]:
    """Compare the runs made so far in a table.

    Returns:
        The outcomes, by name; only those of the runs that have been made.
    """
    outcomes = {name: outcome(run) for name, run in prepared.items()}
    made = {name: result for name, result in outcomes.items() if result is not None}
    if made:
        rows = [
            {
                "Configuration": name,
                "Feasible runs": f"{result.feasible} of {result.runs}",
                "Median ICMOP": round(result.icmop, 3),
            }
            for name, result in made.items()
        ]
        st.markdown("**What you have so far**")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    return made


def _render_introduction(jar: Path) -> None:
    st.markdown(
        "Many real problems have **constraints**: a solution is only valid if it satisfies them, "
        "however good its objectives are. A solution that satisfies them all is **feasible**; "
        "the **violation** of another one says how far it is from that, and it is 0 for a "
        "feasible one.\n\n"
        "In Evolver-Studio those problems are listed apart (*With constraints*) wherever a problem "
        "is chosen, and the algorithms then use their parameter space **for constraints**. In "
        "this tutorial you will:\n\n"
        "- see the techniques that handle constraints;\n"
        "- learn how a run that may find no feasible solution is measured;\n"
        f"- run {ALGORITHM} without a technique and with one on **{SIMPLE_PROBLEM}**, and see that "
        f"on **{HARD_PROBLEM}** choosing the technique matters;\n"
        "- let a training choose it, and follow how it reaches the configurations that find "
        "feasible solutions.\n\n"
        "It is the interactive companion of Evolver's tutorial "
        f"[E12. Problems with constraints]({E12_TUTORIAL_URL}). It assumes the tutorials *Solving "
        "a problem with a configurable algorithm* and *Your first guided training*."
    )
    catalogue = registered_problems(str(jar)) or {}
    constrained = [p for p in catalogue.values() if p.has_constraints]
    shown = [p for p in constrained if p.name in (SIMPLE_PROBLEM, HARD_PROBLEM)]
    if not shown:
        st.info("This needs an Evolver that describes its constraints (2.6 or later).")
        return
    families = sorted({p.family for p in constrained})
    st.markdown(
        f"Evolver has **{len(constrained)} problems with constraints** "
        f"({', '.join(families)}). The two of this tutorial:"
    )
    render_table(shown, len(constrained), f"{KEY_PREFIX}_problems")
    st.caption(
        "The sets of runs of this tutorial take up to 10 seconds each, and its training about "
        f"three minutes with 8 cores. Timings measured on: {TIMINGS_MEASURED_ON}."
    )
    st.page_link("pages/explore_problems.py", label="Open Explore › Problems", icon="🧩")


def _render_techniques(jar: Path) -> None:
    algorithm = next(a for a in BASE_ALGORITHMS if a.name == ALGORITHM)
    file = algorithm.parameter_space_file("Double", constrained=True)
    space = parse_parameter_space(parameter_space_text(jar, file))
    parameter = next(p for p in space if p.name in CONSTRAINT_HANDLING_PARAMETERS)
    assert isinstance(parameter, CategoricalParameter)
    techniques = "\n".join(
        f"- `{choice.value}`: {CONSTRAINT_HANDLING.get(choice.value, '')}"
        for choice in parameter.choices
    )
    st.markdown(
        f"An algorithm that ignores the constraints ranks the solutions only by their objectives. "
        f"`{file}`, the parameter space of {ALGORITHM} for problems with constraints, is its usual "
        f"one plus the parameter `{parameter.name}`, a **technique** that says which of two "
        f"solutions is better when they may be infeasible:\n\n{techniques}\n\n"
        "A technique is a parameter like any other: some have parameters of their own, and a "
        "training can choose it. The other algorithms have theirs (PAES and RVEA with fewer "
        "techniques, and MOEA/D with `subproblemUpdateCriterion`, since it does not rank its "
        "population): **Explore › Base algorithms** shows them."
    )
    answer = st.radio(
        "Quick check: with cdp, which is better, an infeasible solution with excellent objectives "
        "or a feasible one with poor objectives?",
        ["The infeasible one", "The feasible one", "It depends on the violation"],
        index=None,
        key=f"{KEY_PREFIX}_cdp_quiz",
    )
    if answer == "The feasible one":
        st.success("Right: with cdp a feasible solution is always better than an infeasible one.")
    elif answer is not None:
        st.error("Not quite: cdp compares the violations only between two infeasible solutions.")
    st.page_link("pages/explore_base_algorithms.py", label="Open Base algorithms", icon="🧬")


def _render_measuring(jar: Path) -> None:
    st.markdown(
        "A run on a problem with constraints is described by two things: **whether it found "
        "feasible solutions**, and **how good they are**. The usual indicators only tell the "
        "second, and worse: the infeasible solutions may lie beyond the true front, so a run "
        "that found no feasible solution could look better than one that did.\n\n"
        "**ICMOP** (`ConstrainedMultiobjectiveIndicator`) puts both in one value, lower is "
        "better:\n\n"
        "| Value | The run |\n|---|---|\n"
        "| below 0 | has feasible solutions that dominate the nadir point: minus their "
        "hypervolume |\n"
        f"| 0 to {THRESHOLD:g} | has feasible solutions, still far: their distance to the region "
        "of interest |\n"
        f"| above {THRESHOLD:g} | has **none**: {THRESHOLD:g} plus the least violation |\n\n"
        "**CEP** (`ConstrainedEpsilon`) does the same with the epsilon indicator. Run algorithm "
        "and Validation choose these two by default on problems with constraints."
    )
    value = st.slider(
        "Try it: a run with this ICMOP…", -1.0, 3.0, 1.5, step=0.1, key=f"{KEY_PREFIX}_icmop"
    )
    if value > THRESHOLD:
        st.error(
            f"…found **no feasible solution**: its least violating one violates the constraints "
            f"by {value - THRESHOLD:.1f}."
        )
    elif value >= 0:
        st.warning("…found feasible solutions, but none of them dominates the nadir point yet.")
    else:
        st.success(f"…found feasible solutions whose hypervolume is {-value:.1f}.")


def _render_with_and_without(jar: Path) -> None:
    st.markdown(
        f"{ALGORITHM} on **{SIMPLE_PROBLEM}** with its default configuration, {RUNS} runs of "
        f"{EVALUATIONS:,} evaluations, **without a technique** (the parameter space that ignores "
        "the constraints, which Run algorithm does not offer for this problem) and **with cdp**:"
    )
    without, cdp = _without(SIMPLE_PROBLEM), _cdp(SIMPLE_PROBLEM)
    st.markdown("**Without a technique**")
    render_prepared_run(without, jar, can_open=False)
    st.markdown("**With cdp**")
    render_prepared_run(cdp, jar)
    made = _render_comparison({"Without a technique": without, "cdp": cdp})
    if len(made) < 2:
        return
    ignored, handled = made["Without a technique"], made["cdp"]
    if ignored.feasible < handled.feasible:
        st.markdown(
            f"Without a technique, **{ignored.feasible} of {ignored.runs}** runs found feasible "
            f"solutions; with cdp, **{handled.feasible} of {handled.runs}**. The red crosses of "
            "the first chart are infeasible solutions: the search went where the objectives are "
            "better, to the front of the problem without its constraints."
        )
    else:
        st.markdown("Here the technique did not find feasible solutions in more runs.")


def _render_which_technique(jar: Path) -> None:
    st.markdown(
        f"On **{HARD_PROBLEM}** the feasible region is harder to reach. The same default "
        "configuration without a technique and with cdp, and with another technique, **push and "
        "pull** (`pushPull`), which first ignores the constraints to cross the infeasible regions "
        "and then goes back to the feasible one:"
    )
    runs = {
        "Without a technique": _without(HARD_PROBLEM),
        "cdp": _cdp(HARD_PROBLEM),
        "pushPull": _push_pull(jar),
    }
    for name, prepared in runs.items():
        st.markdown(f"**{name}**")
        render_prepared_run(prepared, jar, show_front=False, can_open=name != "Without a technique")
    made = _render_comparison(runs)
    if len(made) < 3:
        st.info("Run the three to compare them.")
        return
    cdp, ignored, push_pull = made["cdp"], made["Without a technique"], made["pushPull"]
    if cdp.feasible <= ignored.feasible:
        st.markdown(
            f"cdp finds feasible solutions in {cdp.feasible} of {cdp.runs} runs, no more than "
            f"ignoring the constraints ({ignored.feasible}): preferring always the feasible "
            "solutions stops the search at the edge of the first infeasible region it meets."
        )
    if push_pull.feasible > cdp.feasible:
        st.markdown(
            f"pushPull does in **{push_pull.feasible} of {push_pull.runs}**: here, which technique "
            "is used matters."
        )
    st.markdown(
        "But this is one value of each of its parameters, with the rest of the configuration as "
        "it was: finding the technique and the configuration that suit a problem is what "
        "**training** is for."
    )


def _render_training(jar: Path) -> None:
    st.markdown(
        "A training on problems with constraints puts feasibility first. The value of a "
        "configuration for each meta-objective is\n\n"
        "> **6k + 2v + Q**\n\n"
        "where **k** is how many of its runs found no feasible solution (*failed runs*), **v**, "
        "below 1, how much they violated the constraints, and **Q** the quality of the feasible "
        "runs, at most 1. Each failed run adds 6, more than the rest can, so the search first "
        "reaches the configurations whose runs do not fail, and then improves their fronts."
    )
    columns = st.columns(3)
    failed = columns[0].slider("Failed runs (k)", 0, 5, 1, key=f"{KEY_PREFIX}_k")
    violation = columns[1].slider("Violation (v)", 0.0, 0.9, 0.2, key=f"{KEY_PREFIX}_v")
    quality = columns[2].slider("Quality (Q)", -1.0, 1.0, -0.5, key=f"{KEY_PREFIX}_q")
    scaled = violation if failed else 0.0
    st.markdown(
        f"This configuration's value is **{6 * failed + 2 * scaled + quality:.2f}**"
        + (" (without failed runs there is no violation to add)." if not failed else ".")
    )
    st.markdown(
        f"The button opens **Training** with a training of {ALGORITHM} for {HARD_PROBLEM}, the "
        f"technique included in its parameter space: {META_OPTIMIZER} as meta-optimizer, "
        f"{META_EVALUATIONS:,} configurations, **ICMOP and EP** as meta-objectives, and the "
        "population written, because the front alone hides the configurations that fail as soon "
        "as one does not. Launch it, follow its tab **Feasibility**, and come back when it ends."
    )
    if st.button("Open Training with this training", icon="🏋️", type="primary"):
        open_training(training_form_state())
    training = latest_tutorial_training()
    if training is not None:
        st.success(f"Your training {training.run_id} has finished: go on to the next step.")


def _render_what_it_found(jar: Path) -> None:
    training = latest_tutorial_training()
    if training is None:
        st.info("You have not run the training yet: launch it in the previous step and come back.")
        return
    render_summary(training)
    state = monitor_state(training)
    st.markdown(
        "**How the search reaches the feasible configurations.** At first most configurations, "
        "which are random, have failed runs; the training replaces them."
    )
    render_feasibility(state, f"{KEY_PREFIX}_feasibility")
    if training.has_population:
        st.markdown(
            "**The population.** The configurations with failed runs are in red, 6 or more above "
            "the others. Untick *Follow the latest checkpoint* to see the first ones."
        )
        render_population(state, f"{KEY_PREFIX}_population")


def best_configuration(training: FinishedTraining) -> str:
    """The configuration of a training's front with the lowest first meta-objective."""
    objective = next(iter(training.configurations[0].objectives))
    return min(training.configurations, key=lambda c: c.objectives[objective]).configuration


def _render_is_it_better(jar: Path) -> None:
    training = latest_tutorial_training()
    if training is None:
        st.info("You have not run the training yet: launch it in step 5 and come back.")
        return
    configuration = best_configuration(training)
    tuned = prepared_run(f"tuned_{training.run_id}", HARD_PROBLEM, None, configuration)
    st.markdown(
        f"The configuration with the lowest ICMOP of your training, against the default one, on "
        f"{HARD_PROBLEM}, {RUNS} runs each:"
    )
    st.code(configuration, language=None, wrap_lines=True)
    st.markdown("**Default (cdp)**")
    render_prepared_run(_cdp(HARD_PROBLEM), jar, show_front=False)
    st.markdown("**Tuned**")
    render_prepared_run(tuned, jar)
    made = _render_comparison({"Default (cdp)": _cdp(HARD_PROBLEM), "Tuned": tuned})
    if len(made) < 2:
        return
    default, found = made["Default (cdp)"], made["Tuned"]
    better = found.icmop < default.icmop
    st.markdown(
        f"Tuned: feasible in **{found.feasible} of {found.runs}** runs, median ICMOP "
        f"**{found.icmop:.3f}**; default: {default.feasible} of {default.runs}, "
        f"{default.icmop:.3f}. "
        + (
            "The tuned one is better, on the problem it was tuned for."
            if better
            else "The tuned one is not better: the training measured each configuration with one "
            "run, and that run may have been a lucky one."
        )
    )
    st.markdown(
        "Whether it works on other problems is what **Validation** tests: choose *With "
        "constraints* there, problems not used in the training, and see the tab **Feasibility** "
        "of its analysis."
    )


def _render_next_steps(jar: Path) -> None:
    st.markdown(
        "**What to try next:**\n\n"
        f"- In **Run algorithm**, run {HARD_PROBLEM} with the front shown while it runs: the "
        "infeasible solutions are in red, and you can see when the population enters the "
        "feasible region.\n"
        "- Change the technique in section 3 of Run algorithm, with other values of its "
        "parameters.\n"
        "- **Train** on several problems (the MW suite), with several runs per configuration, so "
        "that the configuration works on more than one problem.\n"
        "- In **Validation**, compare your tuned configuration with other algorithms on problems "
        "*with constraints*, or on *both* kinds."
    )
    st.page_link("pages/solve.py", label="Open Run algorithm", icon="▶️")
    st.page_link("pages/validation.py", label="Open Validation", icon="✅")
    st.button(
        "All the tutorials",
        icon="🎓",
        on_click=open_tutorial,
        args=(None,),
        key=f"{KEY_PREFIX}_all_tutorials",
    )


STEPS: tuple[TutorialStep, ...] = (
    TutorialStep("What are problems with constraints?", _render_introduction),
    TutorialStep("Step 1: the techniques", _render_techniques),
    TutorialStep("Step 2: measuring a run", _render_measuring),
    TutorialStep("Step 3: with and without a technique", _render_with_and_without),
    TutorialStep("Step 4: the technique matters", _render_which_technique),
    TutorialStep("Step 5: a training puts feasibility first", _render_training),
    TutorialStep("Step 6: what the training found", _render_what_it_found),
    TutorialStep("Step 7: is it better than the default?", _render_is_it_better),
    TutorialStep("What to try next", _render_next_steps),
)
