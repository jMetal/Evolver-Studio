"""Tutorial S2, "Exploring a parameter space": its steps.

The interactive counterpart of Evolver's tutorial E1 (docs/tutorials/parameter_spaces.rst): the
same concepts, on the same parameter spaces (NSGA-II for continuous and binary problems), read
in the tables of the Explore pages instead of through Java code. Version 1.1: the spaces are shown
as the Explore pages show them (a table with an *Active if* column), and the tutorial ends with the
meta-optimizers and the quality indicators, the other pages of Explore.
"""

from pathlib import Path

import streamlit as st

from evolver_studio.catalogue import META_ALGORITHMS, QUALITY_INDICATORS
from evolver_studio.configuration import complete_values, configuration_string
from evolver_studio.parameter_form import (
    render_parameter_space_readonly,
    render_parameter_space_table,
)
from evolver_studio.parameter_space import (
    CategoricalParameter,
    ParameterSpec,
    active_parameter_names,
    active_sub_parameters,
    count_parameters,
    parse_parameter_space,
)
from evolver_studio.resource_files import parameter_space_text
from evolver_studio.tutorial_navigation import open_tutorial
from evolver_studio.tutorials import TutorialStep

E1_TUTORIAL_URL = (
    "https://github.com/jMetal/Evolver/blob/develop/docs/tutorials/parameter_spaces.rst"
)
KEY_PREFIX = "tutorial_parameter_spaces"


def _load(jar: Path, filename: str) -> list[ParameterSpec]:
    return parse_parameter_space(parameter_space_text(jar, filename))


def _render_introduction(jar: Path) -> None:
    st.markdown(
        "Every configurable algorithm in Evolver is described by a **parameter space**: the set "
        "of all the ways it can be configured — which crossover and mutation it uses, how it "
        "selects parents, whether it keeps an external archive, and the settings each of those "
        "components needs.\n\n"
        "A parameter space serves two purposes:\n\n"
        "- **Configuring an algorithm**: choosing a value for every parameter that applies gives a "
        "*configuration*, a concrete variant of the algorithm that can be run.\n"
        "- **Tuning an algorithm**: a meta-optimizer searches the parameter space for the "
        "configurations that perform best.\n\n"
        "In this tutorial you will read the parameter space of NSGA-II, see how its parameters "
        "depend on each other, find out which ones a configuration activates, compare two "
        "encodings of the same algorithm, and visit the other things the **Explore** pages show: "
        "the operators of the meta-optimizers and the quality indicators. It is the interactive "
        f"companion of Evolver's tutorial [E1. Parameter spaces]({E1_TUTORIAL_URL}), which covers "
        "the same ideas with Java code.\n\n"
        "The spaces you will read here are the ones you can browse yourself in **Explore › Base "
        "algorithms**, and the ones the form of **Run algorithm** adjusts when you configure an "
        "algorithm (tutorial S3)."
    )


def _render_reading_the_table(jar: Path) -> None:
    parameters = _load(jar, "NSGAIIDouble.yaml")
    st.markdown(
        "This is the parameter space of NSGA-II for continuous problems (`NSGAIIDouble.yaml`), "
        "as the Explore page shows it: **one row per parameter**.\n\n"
        "- *Parameter*: its name, indented under the parameter it depends on.\n"
        "- *Type*: **categorical** (it takes one of a list of values, e.g. "
        "`createInitialSolutions`), **integer** or **double** (a number in a range, e.g. "
        "`crossoverProbability`).\n"
        "- *Active if*: when it applies (you will see how in the next step); empty for the "
        "**top-level** parameters, which every configuration sets.\n"
        "- *Domain*: its values, or its range `[min, max]`.\n\n"
        "The line above the table counts the parameters. The filter keeps the rows that mention "
        "what you type, together with the parameters they hang from."
    )
    render_parameter_space_table(parameters, key=f"{KEY_PREFIX}_table")
    st.markdown(
        "**Try it:** type `archive` in the filter. Which parameter do the matches hang from?"
    )
    answer = st.radio(
        "Quick check: how many top-level parameters does this space have?",
        ["3", "5", "34"],
        index=None,
        key=f"{KEY_PREFIX}_top_level_quiz",
    )
    if answer == "5":
        st.success("Right: five, and 34 parameters in all, counting every sub-parameter.")
    elif answer is not None:
        st.error("Not quite: look at the summary above the table.")
    with st.expander("The same space as a tree"):
        with st.container(height=400):
            render_parameter_space_readonly(parameters)


def _render_relations(jar: Path) -> None:
    parameters = _load(jar, "NSGAIIDouble.yaml")
    crossover = _find(parameters, "crossover")
    st.markdown(
        "Parameters can have sub-parameters of two kinds, which the *Active if* column of the "
        "table tells apart:\n\n"
        "- `crossover (any)`: a **global sub-parameter**, which applies whatever value its parent "
        "takes. Every crossover has a probability, so `crossoverProbability` is a global "
        "sub-parameter of `crossover`.\n"
        "- `crossover = SBX`: a **conditional parameter**, which applies only when its parent "
        "takes that value. `sbxDistributionIndex` only makes sense for SBX.\n\n"
        "Pick a crossover and see which of its sub-parameters apply:"
    )
    chosen = st.selectbox(
        "crossover", [choice.value for choice in crossover.choices], key=f"{KEY_PREFIX}_crossover"
    )
    active = active_sub_parameters(crossover, {"crossover": chosen})
    st.write("Active sub-parameters:", ", ".join(f"`{p.name}`" for p in active))
    _render_relations_quiz()


def _render_relations_quiz() -> None:
    answer = st.radio(
        "Quick check: which of these parameters applies only when the crossover is `blxAlpha`?",
        ["crossoverProbability", "blxAlphaCrossoverAlpha", "crossoverRepairStrategy"],
        index=None,
        key=f"{KEY_PREFIX}_quiz",
    )
    if answer == "blxAlphaCrossoverAlpha":
        st.success("Right: it is a conditional parameter of `crossover` for the value blxAlpha.")
    elif answer is not None:
        st.error("Not quite: that one is a global sub-parameter, active for every crossover.")


def _render_activation(jar: Path) -> None:
    parameters = _load(jar, "NSGAIIDouble.yaml")
    st.markdown(
        "A parameter is **active** when it applies to the current configuration: top-level "
        "parameters always are, and a sub-parameter is when its parent is active (and, for a "
        "conditional one, when the parent has the right value). Only active parameters need a "
        "value. Choose values below — for instance, switch `algorithmResult` to "
        "`externalArchive` — and watch how the set of active parameters changes:"
    )
    choices = _render_choice_widgets(parameters, {})
    active = active_parameter_names(parameters, choices)
    st.metric("Active parameters", f"{len(active)} of {count_parameters(parameters)}")
    st.write(", ".join(f"`{name}`" for name in active))
    st.markdown(
        "A **configuration** gives a value to each active parameter. Written as Evolver reads it "
        "(`--parameter value`, in the order of the space), the choices above are:"
    )
    st.code(
        configuration_string(parameters, complete_values(parameters, choices)),
        language=None,
        wrap_lines=True,
    )
    st.caption(
        "The numbers are the middle of each range, only so that there is a value to show. This is "
        "what the form of **Run algorithm** shows and adjusts, with a widget for each active "
        "parameter."
    )


def _render_choice_widgets(
    parameters: list[ParameterSpec], choices: dict[str, str]
) -> dict[str, str]:
    for parameter in parameters:
        if isinstance(parameter, CategoricalParameter) and len(parameter.choices) > 1:
            choices[parameter.name] = st.selectbox(
                parameter.name,
                [choice.value for choice in parameter.choices],
                key=f"{KEY_PREFIX}_choice_{parameter.name}",
            )
        _render_choice_widgets(active_sub_parameters(parameter, choices), choices)
    return choices


def _render_encodings(jar: Path) -> None:
    st.markdown(
        "NSGA-II can also solve binary problems. Its parameter space for them, "
        "`NSGAIIBinary.yaml`, has **the same top-level parameters**, but different operators: "
        "SBX or polynomial mutation make no sense on bit strings, so it offers HUX, uniform or "
        "single-point crossover and bit-flip mutation instead. Evolver turns each value into the "
        "right component for the encoding through a *parameter factory*."
    )
    tabs = st.tabs(["Continuous (Double)", "Binary"])
    for tab, filename in zip(tabs, ("NSGAIIDouble.yaml", "NSGAIIBinary.yaml"), strict=True):
        with tab:
            st.caption(filename)
            render_parameter_space_table(_load(jar, filename), key=f"{KEY_PREFIX}_{filename}")


def _render_beyond_algorithms(jar: Path) -> None:
    st.markdown(
        "The algorithms you have read so far are the ones you **run**. Evolver also has "
        "**meta-optimizers**, the algorithms that *search* a parameter space for good "
        "configurations (training), and the page **Explore › Meta-optimizers** shows what each "
        "one is made of. A meta-optimizer is configured through its own operators (crossover, "
        "mutation, selection), and for each *encoding* of the configurations it searches: a flat "
        "vector of numbers, or a derivation tree."
    )
    nsgaii = next(meta for meta in META_ALGORITHMS if meta.name == "NSGA-II")
    flat, tree = st.tabs(["NSGA-II, flat encoding", "NSGA-II, tree encoding"])
    with flat:
        st.caption(nsgaii.operator_parameter_space_file)
        render_parameter_space_table(
            _load(jar, nsgaii.operator_parameter_space_file), key=f"{KEY_PREFIX}_meta_flat"
        )
    with tree:
        st.caption(nsgaii.tree_operator_parameter_space_file)
        render_parameter_space_table(
            _load(jar, nsgaii.tree_operator_parameter_space_file), key=f"{KEY_PREFIX}_meta_tree"
        )
    answer = st.radio(
        "Quick check: which crossover does the tree encoding offer?",
        ["SBX", "subtree", "blxAlpha"],
        index=None,
        key=f"{KEY_PREFIX}_tree_quiz",
    )
    if answer == "subtree":
        st.success("Right: the only one. It crosses derivation trees, so SBX does not apply.")
    elif answer is not None:
        st.error("Not quite: look at the *crossover* row of the tree encoding's table.")
    st.markdown(
        "Training also needs to know what *good* means. The page **Explore › Quality "
        "indicators** lists the measures of a front a training can minimize, and a training run "
        "minimizes two of them:\n\n"
        + "\n".join(
            f"- **{indicator.short_name}**, {indicator.full_name}: {indicator.measures}"
            for indicator in QUALITY_INDICATORS
        )
        + "\n\nThe page **Explore › Problems** will list the problems an algorithm can be run on."
    )


def _render_next_steps(jar: Path) -> None:
    st.markdown(
        "You now know how to read a parameter space. Keep exploring on your own in the Explore "
        "pages, which show each space as a table:\n\n"
        "- In **Base algorithms**, open **MOEA/D** with the Double encoding. Which top-level "
        "parameters does it add to those of NSGA-II?\n"
        "- Open **RVEA**: its `replacement` parameter chooses between three variants of the "
        "algorithm. Which parameters does each one activate?\n"
        "- Open **NSGA-II** with the Permutation encoding. Which operators does it offer?\n"
        "- In **Meta-optimizers**, open **AsyncNSGA-II**: how does its flat catalogue differ from "
        "NSGA-II's?\n\n"
        "**Next:** tutorial **S3** configures one of these algorithms, runs it on a problem and "
        "reads the front it finds; the form you will use shows the parameters you have just "
        "learned to read. For the same concepts with Java code, see Evolver's tutorial "
        f"[E1]({E1_TUTORIAL_URL})."
    )
    st.page_link("pages/explore_base_algorithms.py", label="Explore the base algorithms", icon="🧬")
    st.page_link("pages/explore_meta_optimizers.py", label="Explore the meta-optimizers", icon="🎛️")
    st.page_link(
        "pages/explore_quality_indicators.py", label="See the quality indicators", icon="📏"
    )
    st.button(
        "All the tutorials",
        icon="🎓",
        on_click=open_tutorial,
        args=(None,),
        key=f"{KEY_PREFIX}_all_tutorials",
    )


def _find(parameters: list[ParameterSpec], name: str) -> CategoricalParameter:
    pending = list(parameters)
    while pending:
        parameter = pending.pop(0)
        if isinstance(parameter, CategoricalParameter):
            if parameter.name == name:
                return parameter
            pending.extend(parameter.global_sub_parameters)
            pending.extend(p for c in parameter.choices for p in c.conditional_parameters)
    raise ValueError(f"No categorical parameter named {name}")


STEPS: tuple[TutorialStep, ...] = (
    TutorialStep("What is a parameter space?", _render_introduction),
    TutorialStep("Reading a parameter space", _render_reading_the_table),
    TutorialStep("Global and conditional sub-parameters", _render_relations),
    TutorialStep("Active parameters", _render_activation),
    TutorialStep("The same algorithm, another encoding", _render_encodings),
    TutorialStep("Beyond the algorithms", _render_beyond_algorithms),
    TutorialStep("Explore on your own", _render_next_steps),
)
