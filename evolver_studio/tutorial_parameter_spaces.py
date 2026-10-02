"""Tutorial S2, "Exploring a parameter space": its steps.

The interactive counterpart of Evolver's tutorial E1 (docs/tutorials/parameter_spaces.rst): the
same concepts, on the same parameter spaces (NSGA-II for continuous and binary problems), read
drawn as a tree instead of Java code.
"""

from pathlib import Path

import streamlit as st

from evolver_studio.parameter_form import render_parameter_space_readonly
from evolver_studio.parameter_space import (
    CategoricalParameter,
    ParameterSpec,
    active_parameter_names,
    active_sub_parameters,
    count_parameters,
    parse_parameter_space,
)
from evolver_studio.resource_files import parameter_space_text
from evolver_studio.tutorials import TutorialStep

E1_TUTORIAL_URL = (
    "https://github.com/jMetal/Evolver/blob/develop/docs/tutorials/parameter_spaces.rst"
)
KEY_PREFIX = "tutorial_s2"


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
        "depend on each other, find out which ones a configuration activates, and compare two "
        "encodings of the same algorithm. It is the interactive companion of Evolver's tutorial "
        f"[E1. Parameter spaces]({E1_TUTORIAL_URL}), which covers the same ideas with Java code."
    )


def _render_reading_the_tree(jar: Path) -> None:
    parameters = _load(jar, "NSGAIIDouble.yaml")
    st.markdown(
        "This is the parameter space of NSGA-II for continuous problems (`NSGAIIDouble.yaml`), "
        "drawn as a tree. Each bullet is a parameter, in **bold**:\n\n"
        "- A **categorical** parameter lists its values (e.g. `createInitialSolutions`).\n"
        "- A **double** or **integer** parameter shows its range `[min, max]` (e.g. "
        "`crossoverProbability`).\n"
        "- The first-level bullets are the **top-level** parameters: every configuration sets them."
    )
    st.metric("Top-level parameters", len(parameters))
    with st.container(height=400):
        render_parameter_space_readonly(parameters)


def _render_relations(jar: Path) -> None:
    parameters = _load(jar, "NSGAIIDouble.yaml")
    crossover = _find(parameters, "crossover")
    st.markdown(
        "Parameters can have sub-parameters of two kinds, which the tree marks differently:\n\n"
        "- *always:* introduces **global sub-parameters**, which apply whatever value their parent "
        "takes. Every crossover has a probability, so `crossoverProbability` is a global "
        "sub-parameter of `crossover`.\n"
        "- *if SBX:* introduces **conditional parameters**, which apply only when their parent "
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
    columns = st.columns(2)
    for column, (label, filename) in zip(
        columns,
        (("Continuous (Double)", "NSGAIIDouble.yaml"), ("Binary", "NSGAIIBinary.yaml")),
        strict=True,
    ):
        parameters = _load(jar, filename)
        with column:
            st.markdown(f"**{label}** — `{filename}`")
            st.metric("Parameters", count_parameters(parameters))
            with st.container(height=400):
                render_parameter_space_readonly(parameters)


def _render_next_steps(jar: Path) -> None:
    st.markdown(
        "You now know how to read a parameter space. Keep exploring on your own in the Explore "
        "pages, which show each space as a table: one row per parameter, indented under the "
        "parameter it hangs from, with an *Active if* column for the condition that activates it "
        "and a filter to find parameters by name or value:\n\n"
        "- In **Base algorithms**, open **MOEA/D** with the Double encoding. Which top-level "
        "parameters does it add to those of NSGA-II?\n"
        "- Open **NSGA-II** with the Permutation encoding. Which operators does it offer?\n"
        "- In **Meta-optimizers**, NSGA-II, AGE-MOEA and AsyncNSGA-II have their own parameter "
        "spaces too, one for each encoding they support.\n\n"
        f"For the same concepts with Java code, see Evolver's tutorial [E1]({E1_TUTORIAL_URL})."
    )
    st.page_link("pages/explore_base_algorithms.py", label="Explore the base algorithms", icon="🧬")
    st.page_link("pages/explore_meta_optimizers.py", label="Explore the meta-optimizers", icon="🎛️")


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
    TutorialStep("Reading the tree", _render_reading_the_tree),
    TutorialStep("Global and conditional sub-parameters", _render_relations),
    TutorialStep("Active parameters", _render_activation),
    TutorialStep("The same algorithm, another encoding", _render_encodings),
    TutorialStep("Explore on your own", _render_next_steps),
)
