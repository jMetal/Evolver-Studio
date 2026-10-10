"""Tutorial S4, "Analyzing training results": its steps.

The interactive counterpart of Evolver's tutorial E7
(docs/tutorials/analyzing_training_results.rst):
what a finished training leaves, how to read it and how to choose a configuration, through the
Training analysis page instead of through the files and scripts.

The tutorial reads a real training run that ships with the app (tutorial_data/analysis/): NSGA-II
tuned on ZDT4 by NSGA-II, for 3,000 meta-evaluations, run with Evolver 2.4 and its population
written. It is read with the same views as the Training analysis page, which lists the
trainings of the reader. What the text says about its results is read from it, not written down.
"""

from pathlib import Path

import streamlit as st

from evolver_studio.analysis_view import (
    monitor_state,
    render_configurations,
    render_parameters,
    render_summary,
)
from evolver_studio.monitor_view import (
    render_convergence,
    render_indicator_front,
    render_population,
)
from evolver_studio.training_runs import FinishedTraining, list_finished_trainings
from evolver_studio.tutorial_navigation import open_tutorial
from evolver_studio.tutorials import TutorialStep

E7_TUTORIAL_URL = (
    "https://github.com/jMetal/Evolver/blob/develop/docs/tutorials/analyzing_training_results.rst"
)
DATA_DIRECTORY = Path(__file__).parent / "tutorial_data" / "analysis"
KEY_PREFIX = "tutorial_analysis"


def tutorial_training() -> FinishedTraining:
    """The training run the tutorial reads."""
    return list_finished_trainings(DATA_DIRECTORY)[0]


def _render_introduction(jar: Path) -> None:
    training = tutorial_training()
    st.markdown(
        "A training run ends with a **front of configurations**: each one is a way to set up the "
        "base algorithm, and each is a different compromise between the meta-objectives the "
        "training minimized (here, two quality indicators). Which one to use, and whether to "
        "trust it, is decided by reading what the training left behind. That is what the "
        "**Training analysis** page is for: it lists the trainings you have run and shows, for "
        "one:\n\n"
        "- how it was run and what it took;\n"
        "- how its meta-objectives **converged**;\n"
        "- its **front**, and the meta-optimizer's **population** if it was written;\n"
        "- the **configurations** of its final front, to download or to take to Validation or "
        "to Run algorithm;\n"
        "- what those configurations **have in common**, and where they differ from the "
        "algorithm's default configuration.\n\n"
        "This tutorial reads a real training that ships with Evolver-Studio, with the same views "
        f"as the Training analysis page: **{training.algorithm}** ({training.encoding} "
        f"variables) tuned on **{', '.join(training.problems)}**. It ran with Evolver 2.4 and "
        "wrote its population. To follow it on a training of your own, run one in Training and "
        "open Training analysis; "
        f"Evolver's tutorial [E7]({E7_TUTORIAL_URL}) does the same reading with Java code."
    )


def _render_summary(jar: Path) -> None:
    training = tutorial_training()
    render_summary(training)
    st.markdown(
        "Before the results, look at **what produced them**. The meta-optimizer evaluated "
        f"{training.outcome.meta_evaluations:,} configurations; each evaluation ran "
        f"{training.algorithm} on the training problem with the base-level budget. A training "
        "is only as good as that budget, the problems it saw and the number of "
        "meta-evaluations, so a result is read next to them: the same training with more "
        "meta-evaluations, or tuned on other problems, can find other configurations.\n\n"
        "The *meta population* is how many configurations the meta-optimizer evolves at once, "
        "and the *front* below is made of the best of them."
    )


def _render_convergence(jar: Path) -> None:
    training = tutorial_training()
    state = monitor_state(training)
    summaries = state.front.summaries
    st.markdown(
        "A training writes a **checkpoint** every so many meta-evaluations: the front of "
        "configurations found so far, with their value of each meta-objective (lower is better). "
        "The convergence plot follows the best, the median and the worst of the front over the "
        "checkpoints."
    )
    render_convergence(state, f"{KEY_PREFIX}_convergence")
    if len(summaries) >= 2:
        first, last = summaries[0], summaries[-1]
        changes = ", ".join(
            f"{name} from {first.minimum[name]:.3g} to {last.minimum[name]:.3g}"
            for name in last.minimum
        )
        st.markdown(
            f"Here the best of the front went, between evaluation {first.evaluation:,} and "
            f"{last.evaluation:,}: {changes}. What to look for:\n\n"
            "- **A curve that is still falling at the end** means the training was stopped "
            "early: more meta-evaluations would have found better configurations.\n"
            "- **A flat curve for a long time** means the search had converged: the rest of "
            "the budget bought little.\n"
            "- Try the **logarithmic values**, which show the late progress that the first "
            "checkpoints, far from it, hide."
        )


def _render_front(jar: Path) -> None:
    training = tutorial_training()
    st.markdown(
        "The **front** plots the configurations of each checkpoint in the space of the "
        "meta-objectives. Narrow it with *Show last N fronts* to see how it moved: early "
        "fronts are far from the origin, later ones close in on one another."
    )
    render_indicator_front(training.output_directory / "INDICATORS.csv", training.run_id)
    if training.has_population:
        st.markdown(
            "This training also wrote the meta-optimizer's **population**: the configurations "
            "it evolves, of which the front is only the best. Uncheck *Follow the latest "
            "checkpoint* to go back to an earlier one."
        )
        render_population(monitor_state(training), f"{KEY_PREFIX}_population")
        st.markdown(
            "A population that has collapsed onto the front has little left to explore; one "
            "that is still spread out is still searching. Only trainings launched with *Show "
            "the meta-optimizer's population while it runs* have this view."
        )


def _render_configurations(jar: Path) -> None:
    training = tutorial_training()
    st.markdown(
        f"The training ends with the **{len(training.configurations)} configurations** of its "
        "final front. None is better than another on both meta-objectives: each is a "
        "compromise, and the choice depends on what matters to you. Choose one in the box "
        "below to see it as the `--parameter value` pairs that a solve request and Validation "
        "take."
    )
    render_configurations(training, jar, hand_off=False)
    st.markdown(
        "A configuration is the result of one training on one set of problems, with the "
        "randomness of one search. **Do not trust it yet**: Validation runs it many times next "
        "to other algorithms and tests whether the difference is real."
    )


def _render_parameters(jar: Path) -> None:
    training = tutorial_training()
    st.markdown(
        "Which parameters did the training care about? The table has one row per parameter the "
        "front's configurations give a value to. The highlighted rows are where the front "
        "moves away from the default configuration of the algorithm: here, for instance, the "
        "training returns an external archive (`algorithmResult`) where the default "
        "configuration returns its population."
    )
    render_parameters(training, jar)
    st.markdown(
        f"With only {len(training.configurations)} configurations on the front, read the "
        "agreement as a hint, not as a measure; a longer training gives a larger front and a "
        "firmer reading. Choose a parameter below the table to see how the configurations "
        "spread over its values, with the default one marked."
    )


def _render_next_steps(jar: Path) -> None:
    training = tutorial_training()
    st.markdown(
        "**Taking a configuration on.** In the Training analysis page, the *Configurations* tab "
        "of any of your trainings has three buttons for the configuration chosen:\n\n"
        "- **Download**, as a `.txt` of `--parameter value` pairs;\n"
        "- **Validate it**, which opens *Validation* with it as the tuned configuration, "
        "compared with the algorithm's default one, on the problems you choose there (those of "
        "the training, and some it never saw);\n"
        "- **Run it**, which opens *Run algorithm* with it, on a problem of the training.\n\n"
        "The *Best configurations* tab of Training offers *Validate this configuration* too, "
        "even while the training is running."
    )
    st.markdown("As an example, this is what the buttons do for the training of the tutorial:")
    render_configurations(training, jar, hand_off=True)
    st.markdown(
        "**What an analysis does not tell.** It describes one training: how stable its result "
        "is across repeated trainings, and how the configurations do on problems it did not "
        "see, are questions for *Validation*, and for several trainings with different seeds."
    )
    st.page_link("pages/training_analysis.py", label="Open Training analysis", icon="🔍")
    st.button(
        "All the tutorials",
        icon="🎓",
        on_click=open_tutorial,
        args=(None,),
        key=f"{KEY_PREFIX}_all_tutorials",
    )


STEPS: tuple[TutorialStep, ...] = (
    TutorialStep("What does a training leave?", _render_introduction),
    TutorialStep("Step 1: how it was run, and what it took", _render_summary),
    TutorialStep("Step 2: did it converge?", _render_convergence),
    TutorialStep("Step 3: the front and the population", _render_front),
    TutorialStep("Step 4: choosing a configuration", _render_configurations),
    TutorialStep("Step 5: what the configurations have in common", _render_parameters),
    TutorialStep("Taking a configuration on, and what to try", _render_next_steps),
)
