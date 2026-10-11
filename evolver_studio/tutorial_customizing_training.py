"""Tutorial "Customizing a training run": its steps.

Tutorial "Your first guided training" runs Evolver's E3 training with the Training page's
defaults; this one goes through what else that page lets you choose, and why: the training set
(E6), the meta-objectives and how each configuration is measured (E6), the cost of it all, the
parameter space (E5), the meta-optimizer, its encoding and its budget (E11, E13, E14), and other
encodings of the base algorithm (E10). Each step can open Training with an example of what it
explains, for the reader to launch or adjust; nothing runs in the tutorial itself.

What the text says about Evolver (the parameters of a space, the meta-optimizers that have a tree
encoding, the problems of an encoding, the indicators) is read from the jar and the catalogue.
"""

from dataclasses import replace
from pathlib import Path

import streamlit as st

from evolver_studio.app_state import registered_problems
from evolver_studio.catalogue import BASE_ALGORITHMS, META_ALGORITHMS
from evolver_studio.parameter_space import (
    CategoricalChoice,
    CategoricalParameter,
    ParameterSpec,
    count_parameters,
    parse_parameter_space,
    with_selected_choices,
)
from evolver_studio.problem_catalogue import problems_with_encoding
from evolver_studio.resource_files import parameter_space_text
from evolver_studio.training_choices import training_indicators
from evolver_studio.training_presets import open_training, training_state
from evolver_studio.tutorial_navigation import open_tutorial
from evolver_studio.tutorials import TutorialStep

EVOLVER_TUTORIALS = "https://github.com/jMetal/Evolver/blob/develop/docs/tutorials/"
E5_URL = f"{EVOLVER_TUTORIALS}designing_parameter_spaces.rst"
E6_URL = f"{EVOLVER_TUTORIALS}training_sets_indicators_budgets.rst"
E10_URL = f"{EVOLVER_TUTORIALS}binary_and_permutation_encodings.rst"
E11_URL = f"{EVOLVER_TUTORIALS}budgets.rst"
E14_URL = f"{EVOLVER_TUTORIALS}tree_versus_flat_encoding.rst"
KEY_PREFIX = "tutorial_customizing"
ALGORITHM = "NSGA-II"
# E3's training took about 3 minutes for 24 million base-level evaluations of ZDT4, on 8 cores.
MEASURED_EVALUATIONS = 24e6
MEASURED_MINUTES = 3.0
TIMINGS_MEASURED_ON = "Apple M5 Pro, 8 cores used by the training, Java 21"
TRAINING_SET = ("ZDT1", "ZDT4", "ZDT6")


def _preset_button(label: str, state: dict[str, object], key: str) -> None:
    if st.button(label, icon="🏋️", key=f"{KEY_PREFIX}_{key}"):
        open_training(state)


def _render_introduction(jar: Path) -> None:
    st.markdown(
        "Tutorial *Your first guided training* ran one training, E3's, with the defaults of the "
        "**Training** page: NSGA-II tuned for ZDT4, NHV and EP as meta-objectives, each "
        "configuration run once, NSGA-II as meta-optimizer with the flat encoding, 2000 "
        "configurations. Every one of those choices can be changed, and each changes what the "
        "training finds and what it costs:\n\n"
        "1. the **training set**: the problems the configuration must be good at;\n"
        "2. the **meta-objectives** and how each configuration is **measured**;\n"
        "3. what it all **costs**;\n"
        "4. the **parameter space**: what the training may change;\n"
        "5. the **meta-optimizer**, its **encoding** and its **budget**;\n"
        "6. other **encodings** of the base algorithm: binary and permutation problems.\n\n"
        "Each step has a button that opens Training with an example, for you to look at, launch "
        "or adjust. Evolver's tutorials "
        f"[E5]({E5_URL}), [E6]({E6_URL}), [E10]({E10_URL}), [E11]({E11_URL}) and "
        f"[E14]({E14_URL}) cover the same ground with Java code."
    )


def _render_training_set(jar: Path) -> None:
    catalogue = registered_problems(str(jar)) or {}
    rows = [
        f"- **{name}**: {catalogue[name].number_of_objectives} objectives, "
        f"{catalogue[name].number_of_variables} variables"
        for name in TRAINING_SET
        if name in catalogue
    ]
    st.markdown(
        "A configuration tuned for one problem is good at *that* problem; one tuned for several "
        "must be good at all of them, and is more likely to be good at the next. In Training, "
        "the **Training problems** selector (or *Browse the problems*) chooses them, and the "
        "table under it gives each one:\n\n"
        "- its **arguments** (DTLZ2 with 12 variables and 2 objectives is `12, 2`);\n"
        "- its **reference front**, against which its indicators are computed;\n"
        "- its **evaluations**, the budget of every run on it: a problem that is harder may "
        "need more.\n\n"
        "Every problem of a training set must have the base algorithm's encoding. The example "
        "trains NSGA-II on three problems of the ZDT family:\n\n" + "\n".join(rows)
    )
    st.markdown(
        "Each configuration is now run on all three: the training costs three times E3's, and "
        "each meta-objective of a configuration is the mean of its values on the three problems."
    )
    _preset_button(
        "Open Training with this training set",
        training_state(problems=TRAINING_SET),
        "training_set",
    )


def _render_meta_objectives(jar: Path) -> None:
    indicators = training_indicators()
    st.markdown(
        "The **meta-objectives** are the quality indicators the meta-optimizer minimizes for "
        "each configuration; Training offers the ones Evolver computes:\n\n"
        + "\n".join(f"- **{i.registry_name}** ({i.short_name}): {i.measures}" for i in indicators)
        + "\n\nThey come in pairs that complement each other (Evolver's E6 discusses them):\n\n"
        "- **NHV and EP**, the default: NHV measures convergence and spread together; EP tells "
        "apart the configurations whose fronts are all far from the reference point, which NHV "
        "gives 1.\n"
        "- **IGD+ and EP**: IGD+ as the main objective instead of NHV.\n"
        "- **EP and Spread**: convergence and how evenly the front is spread (Spread only for two "
        "objectives; GeneralizedSpread for more).\n"
        "- **HypervolumeMinus and EP**: the hypervolume needs only a reference point, for "
        "problems without a good reference front."
    )
    st.markdown(
        "**How each configuration is measured.** With one run per configuration (the default), "
        "a configuration can win by luck: tutorial *Your first guided training* may have shown "
        "you one. **Independent runs per configuration** measure each one several times, with "
        "different seeds, and take the median of each indicator over them (then the mean over "
        "the problems): more reliable, and as many times more costly. "
        "The **base population size** is that of the base algorithm in every run (MOEA/D and "
        "RVEA need a weight vector file for it)."
    )
    _preset_button(
        "Open Training with IGD+ and EP, and 3 runs per configuration",
        training_state(
            indicators=("InvertedGenerationalDistancePlus", "Epsilon"), independent_runs=3
        ),
        "measure",
    )


def _render_cost(jar: Path) -> None:
    st.markdown(
        "Every configuration the meta-optimizer evaluates runs the base algorithm once per "
        "independent run on every training problem. So the base-level evaluations of a training "
        "are:\n\n"
        "> meta-evaluations × independent runs × (evaluations of each problem, added up)\n\n"
        "Try it:"
    )
    columns = st.columns(4)
    meta = columns[0].number_input(
        "Meta-evaluations", 100, value=2000, step=100, key=f"{KEY_PREFIX}_meta"
    )
    runs = columns[1].number_input("Runs per configuration", 1, value=1, key=f"{KEY_PREFIX}_runs")
    problems = columns[2].number_input("Problems", 1, value=1, key=f"{KEY_PREFIX}_problems")
    evaluations = columns[3].number_input(
        "Evaluations per problem", 1000, value=12000, step=1000, key=f"{KEY_PREFIX}_evaluations"
    )
    total = meta * runs * problems * evaluations
    minutes = MEASURED_MINUTES * total / MEASURED_EVALUATIONS
    st.metric("Base-level evaluations", f"{total / 1e6:,.0f} million")
    st.markdown(
        f"E3's training (2000 × 1 × 1 × 12000 = 24 million) took about {MEASURED_MINUTES:.0f} "
        f"minutes; at the same pace this one would take about **{minutes:,.0f} minutes**. "
        "The pace depends on the problems, the algorithm, the configurations tried and the "
        "**Number of cores**, so take it as an order of magnitude. When the budget is the "
        "constraint, stop the meta-optimizer by **computing time** instead (step 5)."
    )
    st.caption(f"Measured on: {TIMINGS_MEASURED_ON}.")


def _find(parameters: list[ParameterSpec], name: str) -> CategoricalParameter | None:
    for parameter in parameters:
        if isinstance(parameter, CategoricalParameter):
            if parameter.name == name:
                return parameter
            nested = [
                *parameter.global_sub_parameters,
                *(p for choice in parameter.choices for p in choice.conditional_parameters),
            ]
            found = _find(list(nested), name)
            if found is not None:
                return found
    return None


def _restricted(parameters: list[ParameterSpec], name: str, keep: set[str]) -> list[ParameterSpec]:
    """The space with the categorical parameter `name`, wherever it is, keeping only `keep`."""
    result = []
    for parameter in parameters:
        if isinstance(parameter, CategoricalParameter):
            if parameter.name == name:
                parameter = with_selected_choices(parameter, keep)
            parameter = replace(
                parameter,
                global_sub_parameters=tuple(
                    _restricted(list(parameter.global_sub_parameters), name, keep)
                ),
                choices=tuple(
                    CategoricalChoice(
                        choice.value,
                        tuple(_restricted(list(choice.conditional_parameters), name, keep)),
                    )
                    for choice in parameter.choices
                ),
            )
        result.append(parameter)
    return result


def _render_parameter_space(jar: Path) -> None:
    algorithm = next(a for a in BASE_ALGORITHMS if a.name == ALGORITHM)
    parameters = parse_parameter_space(parameter_space_text(jar, algorithm.encodings["Double"]))
    total = count_parameters(parameters)
    st.markdown(
        f"The training searches the **parameter space** of the base algorithm: {ALGORITHM}'s "
        f"has **{total} parameters**, which the flat encoding turns into {total} variables. A "
        "smaller space is searched faster and better, but it cannot find what it leaves out; "
        "a larger one can find more, at a cost. Evolver's tutorial "
        f"[E5. Designing your own parameter space]({E5_URL}) studies it.\n\n"
        "In Training, the expander **Base algorithm parameter space** edits it, in two modes: "
        "**Guided**, a form to remove choices of a categorical parameter and narrow the range "
        "of a numeric one; and **Expert**, the YAML itself. Try what removing choices does, here "
        "with the crossover:"
    )
    crossover = _find(parameters, "crossover")
    if crossover is not None:
        values = [choice.value for choice in crossover.choices]
        keep = st.multiselect(
            "Crossover operators to keep", values, default=values, key=f"{KEY_PREFIX}_crossover"
        )
        if keep:
            reduced = count_parameters(_restricted(parameters, "crossover", set(keep)))
            st.metric(
                "Parameters (variables of the flat encoding)",
                reduced,
                delta=reduced - total if reduced != total else None,
                delta_color="inverse",
            )
            st.caption(
                "Each crossover brings its own sub-parameters: removing it removes them too. "
                "The same edit in Training's guided editor gives the training this space."
            )
        else:
            st.warning("A categorical parameter needs at least one choice.")
    _preset_button(
        "Open Training to edit the parameter space",
        training_state(),
        "space",
    )


def _render_meta_optimizer(jar: Path) -> None:
    wired = [m for m in META_ALGORITHMS if m.wired_into_cli_runner]
    with_tree = [m.name for m in wired if m.supports_tree and m.tree_example_config_file]
    st.markdown(
        f"**The meta-optimizer.** Training offers {', '.join(m.name for m in wired)}. They "
        "differ in how they search (Explore › Meta-optimizers lists their operators, which the "
        "expander *Meta-optimizer operator flags* edits); Random Search, which samples "
        "configurations at random, is the baseline to beat.\n\n"
        f"**Its encoding.** {', '.join(with_tree)} also search the **tree** encoding: a "
        "derivation tree of the parameter space's grammar, with no inactive parameters, where "
        "the flat one is a vector with a variable for every parameter, active or not. Evolver's "
        f"[E14]({E14_URL}) compares them.\n\n"
        "**Its budget.** By **evaluations**, the run is the same on any machine; by **computing "
        "time**, it takes the time you give it, and the number of configurations it tries "
        f"depends on the machine and the cores ([E11]({E11_URL})).\n\n"
        "The example: NSGA-II with the tree encoding, for 5 minutes."
    )
    _preset_button(
        "Open Training with the tree encoding and 5 minutes",
        training_state(meta_encoding="Tree", meta_minutes=5.0),
        "meta",
    )


def _render_encodings(jar: Path) -> None:
    catalogue = registered_problems(str(jar)) or {}
    binary = problems_with_encoding(catalogue, "Binary")
    permutation = problems_with_encoding(catalogue, "Permutation")

    def offering(encoding: str) -> str:
        return ", ".join(
            a.name for a in BASE_ALGORITHMS if a.runnable_today and encoding in a.runnable_encodings
        )

    st.markdown(
        "The base algorithm's **encoding** is that of its problems. Besides real numbers, "
        f"Evolver has **binary** problems ({', '.join(binary[:5])}…) and **permutation** ones "
        f"({', '.join(permutation[:3])}…). In Training, {offering('Binary')} have a binary "
        f"encoding, and {offering('Permutation')} a permutation one. Each encoding has its own "
        "parameter space, with its own operators (bit-flip mutation, permutation crossovers…). "
        f"Evolver's [E10. Binary and permutation encodings]({E10_URL}) tunes NSGA-II for ZDT5 "
        "and for a bi-objective TSP.\n\n"
        "The example: NSGA-II with the binary encoding, tuned for ZDT5."
    )
    _preset_button(
        "Open Training with NSGA-II on ZDT5 (binary)",
        training_state(encoding="Binary", problems=("ZDT5",)),
        "binary",
    )


def _render_next_steps(jar: Path) -> None:
    st.markdown(
        "**Putting it together.** A training worth validating usually has a training set of "
        "several problems of the family you care about, NHV and EP (or IGD+ and EP), a few runs "
        "per configuration if the problems are noisy, and a budget by time that fits your "
        "machine. Then read it in **Training analysis**, and put the configuration it found to "
        "the test in **Validation**: tutorials *Analyzing training results* and *Validating a "
        "configuration*."
    )
    st.page_link("pages/training.py", label="Open Training", icon="🏋️")
    st.button(
        "All the tutorials",
        icon="🎓",
        on_click=open_tutorial,
        args=(None,),
        key=f"{KEY_PREFIX}_all_tutorials",
    )


STEPS: tuple[TutorialStep, ...] = (
    TutorialStep("What can be customized?", _render_introduction),
    TutorialStep("Step 1: the training set", _render_training_set),
    TutorialStep(
        "Step 2: meta-objectives, and measuring each configuration", _render_meta_objectives
    ),
    TutorialStep("Step 3: what it costs", _render_cost),
    TutorialStep("Step 4: the parameter space", _render_parameter_space),
    TutorialStep("Step 5: the meta-optimizer, its encoding and its budget", _render_meta_optimizer),
    TutorialStep("Step 6: binary and permutation problems", _render_encodings),
    TutorialStep("Putting it together", _render_next_steps),
)
