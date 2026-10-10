"""Tutorial "Your first guided training": its steps.

The interactive counterpart of Evolver's tutorial E3
(docs/tutorials/meta_optimization_workflow.rst): the same training (NSGA-II tuned for ZDT4 by
NSGA-II, with NHV and EP as meta-objectives, 2000 configurations), through the Training page
instead of through Java code. The tutorial explains
each piece, fills the Training form with them for the reader to launch and follow there, reads
what the training found, and compares a configuration of its front with the default NSGA-II on
ZDT4.

Until the reader has run one, the steps that read a result use the training that ships with the
app for tutorial "Analyzing training results" (the same setting, 3000 configurations), and say so.
"""

from pathlib import Path

import streamlit as st

from evolver_studio import evolver_client
from evolver_studio.analysis_view import monitor_state, render_configurations, render_summary
from evolver_studio.catalogue import BASE_ALGORITHMS
from evolver_studio.monitor_view import render_convergence
from evolver_studio.parameter_space import count_parameters, parse_parameter_space
from evolver_studio.resource_files import parameter_space_text
from evolver_studio.solve_results import read_indicators
from evolver_studio.training_runs import FinishedTraining, list_finished_trainings
from evolver_studio.tutorial_analysis import tutorial_training
from evolver_studio.tutorial_navigation import open_tutorial
from evolver_studio.tutorial_solving import (
    PreparedRun,
    output_directory,
    remembered_run,
    render_prepared_run,
)
from evolver_studio.tutorials import TutorialStep

E3_TUTORIAL_URL = (
    "https://github.com/jMetal/Evolver/blob/develop/docs/tutorials/meta_optimization_workflow.rst"
)
KEY_PREFIX = "tutorial_first_training"
ALGORITHM = "NSGA-II"
PROBLEM = "ZDT4"
REFERENCE_FRONT = "resources/referenceFronts/ZDT4.csv"
POPULATION_SIZE = 100
BASE_EVALUATIONS = 12000
META_EVALUATIONS = 2000
META_POPULATION_SIZE = 50
UPDATE_EVERY = 100
MINUTES_MEASURED = 3
TIMINGS_MEASURED_ON = (
    "Apple M5 Pro (18 cores, 8 of them used by the training), 64 GB of RAM, macOS 26.6.2, Java "
    "21.0.12 (Oracle JDK)"
)
COMPARISON_RUNS = 5
DEFAULT_ON_ZDT4 = PreparedRun(
    "training_default",
    ALGORITHM,
    "NSGAIIDoubleDefault.txt",
    None,
    PROBLEM,
    REFERENCE_FRONT,
    POPULATION_SIZE,
    BASE_EVALUATIONS,
    COMPARISON_RUNS,
    {},
)


def training_form_state() -> dict[str, object]:
    """The values of the Training form for this tutorial's training, by widget key.

    Most are the page's defaults; the tutorial sets them all, so that the page shows this
    training whatever was chosen before, and asks for the meta-optimizer's population to be
    written, for the Population tab of the monitor.
    """
    return {
        "train_base_algorithm": ALGORITHM,
        f"train_encoding_{ALGORITHM}": "Double",
        "train_problems_Double": [PROBLEM],
        "train_meta_algorithm": ALGORITHM,
        f"train_meta_encoding_{ALGORITHM}": "Flat",
        "train_meta_stop_by": "Evaluations",
        "train_meta_evals": META_EVALUATIONS,
        f"train_meta_population_{ALGORITHM}": META_POPULATION_SIZE,
        f"train_update_every_{META_POPULATION_SIZE}": UPDATE_EVERY,
        "train_write_population": True,
    }


def latest_tutorial_training() -> FinishedTraining | None:
    """The most recent finished training of this tutorial's setting, if the reader has run one."""
    for training in list_finished_trainings(evolver_client.WORKING_DIRECTORY):
        if (
            training.algorithm == ALGORITHM
            and training.encoding == "Double"
            and training.problems == (PROBLEM,)
            and training.meta_algorithm == ALGORITHM
        ):
            return training
    return None


def _training() -> tuple[FinishedTraining, bool]:
    """The training the steps read: the reader's, or the one that ships with the app."""
    own = latest_tutorial_training()
    return (own, True) if own is not None else (tutorial_training(), False)


def _render_introduction(jar: Path) -> None:
    st.markdown(
        "Tutorial *Solving a problem* showed that the configuration of an algorithm matters a "
        "lot. A **training** finds a good one automatically: another multi-objective "
        "algorithm, the **meta-optimizer**, searches the parameter space of the algorithm to "
        "configure, and judges each configuration by running the algorithm with it."
    )
    st.code(
        f"Meta-optimizer ({ALGORITHM})\n"
        "  proposes configurations of the base-level algorithm\n"
        f"    └─> Base-level algorithm ({ALGORITHM}, its parameter space)\n"
        f"          runs each configuration on the training problems ({PROBLEM})\n"
        "            └─> Quality indicators of the fronts found (NHV, EP)\n"
        "                  are the objectives the meta-optimizer minimizes",
        language=None,
    )
    st.markdown(
        f"In this tutorial you will set up the training of Evolver's tutorial "
        f"[E3. Meta-optimization workflow]({E3_TUTORIAL_URL}) in the **Training** page, launch "
        "it and follow it there, read what it found, and compare a configuration it found with "
        f"the default {ALGORITHM}. The training takes about {MINUTES_MEASURED} minutes."
    )
    st.caption(f"Timings measured on: {TIMINGS_MEASURED_ON}.")


def _render_base_level(jar: Path) -> None:
    algorithm = next(a for a in BASE_ALGORITHMS if a.name == ALGORITHM)
    parameters = count_parameters(
        parse_parameter_space(parameter_space_text(jar, algorithm.encodings["Double"]))
    )
    st.markdown(
        f"The **base level** is the algorithm to configure and the problems it is tuned for, "
        "the first sections of the Training page:\n\n"
        f"- **{ALGORITHM}** with real-number variables (*Double*), population {POPULATION_SIZE}. "
        f"Its parameter space has **{parameters} parameters** (tutorial *Exploring a parameter "
        "space*): that is what the training searches.\n"
        f"- The **training set**: **{PROBLEM}** alone, a problem with many local fronts on which "
        f"the default {ALGORITHM} struggles. Each configuration is run on it once, with "
        f"**{BASE_EVALUATIONS:,} evaluations**, and measured against its reference front. A "
        "training set of several problems, and several runs per configuration, give a more "
        "general and more reliable configuration, at a higher cost.\n"
        "- The **quality indicators** of each front are the **meta-objectives**, which the "
        "meta-optimizer minimizes: **NHV**, the normalized hypervolume, is the main one, as it "
        "accounts for both the convergence and the spread of a front; **EP**, the epsilon "
        "indicator, helps early in the search, when many configurations get NHV = 1 and only EP "
        "tells them apart."
    )


def _render_meta_optimizer(jar: Path) -> None:
    runs = META_EVALUATIONS
    st.markdown(
        f"The **meta-optimizer** is also **{ALGORITHM}**, with the **flat** encoding: each "
        "configuration is a vector of numbers in [0, 1], one per parameter (the *tree* "
        "encoding, which some meta-optimizers offer, follows the structure of the parameter "
        "space instead). It evolves a population of "
        f"**{META_POPULATION_SIZE}** configurations and stops after **{META_EVALUATIONS:,} "
        "meta-evaluations**, configurations evaluated.\n\n"
        f"So the training runs {ALGORITHM} {runs:,} times, "
        f"**{runs * BASE_EVALUATIONS / 1e6:.0f} million evaluations** of {PROBLEM}: that is why "
        "it evaluates the configurations in parallel, on the **cores** you give it, and why the "
        "base-level budget matters so much.\n\n"
        "**Evaluations or time.** The meta-optimizer can also stop after a computing time. "
        "With evaluations, the run is the same on any machine; with time, it takes the time you "
        "decide, but how many configurations it tries depends on the machine.\n\n"
        f"Every **{UPDATE_EVERY} meta-evaluations** (a multiple of the population size) the "
        "training writes a checkpoint: the front of configurations found so far, which the "
        "Training page shows as it runs."
    )


def _render_launch(jar: Path) -> None:
    st.markdown(
        "The button below opens **Training** with this training in its form (and *Show the "
        "meta-optimizer's population while it runs* ticked). Check the **Number of cores**, "
        "then press **Launch training**. While it runs, the page has a tab for each question:\n\n"
        "- **Overview**: how far it is, its pace and the time left, and the best values so "
        "far;\n"
        "- **Front**: the front of configurations at each checkpoint;\n"
        "- **Population**: the whole population of the meta-optimizer, and its front over it;\n"
        "- **Convergence**: how the best, the median and the worst of the front improve;\n"
        "- **Best so far**: the configurations of the latest front, to copy or validate;\n"
        "- **Log**: what Evolver prints.\n\n"
        "The training goes on if you leave the page or close the tab; come back to this "
        "tutorial when it has finished."
    )
    if st.button("Open Training with this training", icon="🏋️", type="primary"):
        st.session_state.pop("training_set_rows", None)
        st.session_state.update(training_form_state())
        st.switch_page("pages/training.py")
    own = latest_tutorial_training()
    if own is not None:
        st.success(f"Your training {own.run_id} has finished: go on to the next step.")


def _render_what_it_found(jar: Path) -> None:
    training, own = _training()
    if not own:
        st.info(
            "You have not run the training yet: this step reads the one that ships with the "
            "app, the same setting with 3,000 meta-evaluations. Run yours and come back."
        )
    render_summary(training)
    st.markdown(
        "**Did it converge?** The best, the median and the worst of the front at each checkpoint:"
    )
    render_convergence(monitor_state(training), f"{KEY_PREFIX}_convergence")
    front = training.configurations
    best_nhv = min(front, key=lambda c: c.objectives.get("NHV", float("inf")))
    st.markdown(
        f"The training ends with **{len(front)} configurations** on its front, each a "
        "different compromise between NHV and EP. The one with the lowest NHV has "
        + ", ".join(f"{name} = {value:.4g}" for name, value in best_nhv.objectives.items())
        + ". Choose one to read it, download it, or take it on:"
    )
    render_configurations(training, jar, hand_off=False)


def _render_is_it_better(jar: Path) -> None:
    training, own = _training()
    front = training.configurations
    chosen = min(front, key=lambda c: c.objectives.get("NHV", float("inf")))
    tuned = PreparedRun(
        f"training_tuned_{training.run_id}",
        ALGORITHM,
        None,
        chosen.configuration,
        PROBLEM,
        REFERENCE_FRONT,
        POPULATION_SIZE,
        BASE_EVALUATIONS,
        COMPARISON_RUNS,
        {},
    )
    st.markdown(
        f"Is the configuration with the lowest NHV better than the default {ALGORITHM}? Run "
        f"both on {PROBLEM}, with the training's budget, {COMPARISON_RUNS} times each (seed 1):"
    )
    st.markdown(f"**The default {ALGORITHM}**")
    render_prepared_run(DEFAULT_ON_ZDT4, jar)
    st.markdown(f"**The configuration found** ({'your training' if own else 'the shipped one'})")
    render_prepared_run(tuned, jar)
    default_run, tuned_run = remembered_run(DEFAULT_ON_ZDT4), remembered_run(tuned)
    if default_run is None or tuned_run is None:
        return
    default = read_indicators(output_directory(default_run)).median(numeric_only=True)
    found = read_indicators(output_directory(tuned_run)).median(numeric_only=True)
    measured = chosen.objectives.get("NHV")
    better = found["NHV"] < default["NHV"]
    st.markdown(
        f"Median NHV over the {COMPARISON_RUNS} runs: **{found['NHV']:.4f}** for the "
        f"configuration found against **{default['NHV']:.4f}** for the default"
        + (": better, on the problem it was tuned for." if better else ": not better.")
    )
    if measured is not None and found["NHV"] > 2 * measured:
        st.warning(
            f"The training measured NHV = {measured:.4f} for this configuration, but over "
            f"{COMPARISON_RUNS} runs with other seeds its median is {found['NHV']:.4f}. The "
            "training ran each configuration **once**, and that run was a lucky one: on a "
            f"problem with many local fronts like {PROBLEM}, one run can land near the true "
            "front or get stuck. A configuration chosen by one run each may only be a lucky "
            "one, which is why a training can run each configuration several times "
            "(*independent runs*), at a cost, and why a configuration is validated before it "
            "is trusted."
        )
    st.markdown(
        "Five runs on the problem it was tuned for are not a proof either way: whether it is "
        "better, and on which problems, is what **Validation** tests (tutorial *Validating a "
        "configuration*)."
    )


def _render_next_steps(jar: Path) -> None:
    st.markdown(
        "**What to try next**, in Training:\n\n"
        "- Train on **several problems** (add ZDT1 and ZDT6): the configuration found must "
        "work on all of them, so it generalizes better.\n"
        "- Stop by **computing time** instead of evaluations.\n"
        "- Choose another **meta-optimizer**, or the **tree** encoding (Explore › "
        "Meta-optimizers says which have it).\n"
        "- Use **IGD+ and EP** as meta-objectives.\n\n"
        "And go on reading what a training leaves, in **Training analysis**: the tutorial "
        "*Analyzing training results* does it step by step."
    )
    st.page_link("pages/training_analysis.py", label="Open Training analysis", icon="📊")
    st.button(
        "Go on to Analyzing training results",
        icon="📊",
        on_click=open_tutorial,
        args=("analyzing_training_results",),
        key=f"{KEY_PREFIX}_next_tutorial",
    )
    st.button(
        "All the tutorials",
        icon="🎓",
        on_click=open_tutorial,
        args=(None,),
        key=f"{KEY_PREFIX}_all_tutorials",
    )


STEPS: tuple[TutorialStep, ...] = (
    TutorialStep("What is a training?", _render_introduction),
    TutorialStep("Step 1: the base level", _render_base_level),
    TutorialStep("Step 2: the meta-optimizer", _render_meta_optimizer),
    TutorialStep("Step 3: launch it, and follow it", _render_launch),
    TutorialStep("Step 4: what it found", _render_what_it_found),
    TutorialStep("Step 5: is it better than the default?", _render_is_it_better),
    TutorialStep("What to try next", _render_next_steps),
)
