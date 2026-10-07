"""Tutorials: interactive, step-by-step tutorials, grouped by level."""

from pathlib import Path

import streamlit as st

from evolver_studio import tutorial_parameter_spaces, tutorial_solving, tutorial_validation
from evolver_studio.app_state import render_sidebar, require_evolver_jar
from evolver_studio.tutorial_navigation import (
    SELECTED_KEY,
    STEP_KEY,
    go_to_step,
    open_tutorial,
)
from evolver_studio.tutorials import (
    EVOLVER_TUTORIALS_URL,
    TUTORIALS,
    Tutorial,
    TutorialLevel,
    tutorials_by_level,
)

# Steps of every tutorial with content in this app, by topic (see Tutorial.available).
TUTORIAL_STEPS = {
    "parameter_spaces": tutorial_parameter_spaces.STEPS,
    "solving": tutorial_solving.STEPS,
    "validating_a_configuration": tutorial_validation.STEPS,
}


def _render_catalogue() -> None:
    st.markdown(
        "Step-by-step tutorials on both uses of Evolver-Studio: solving problems with Evolver's "
        "configurable algorithms, and meta-optimization. Each one pairs with the "
        f"[tutorials of Evolver's documentation]({EVOLVER_TUTORIALS_URL}) (E1, E2…) that cover "
        "the same ground with Java code. The ones still to be written are listed without a "
        "number."
    )
    for level in TutorialLevel:
        st.subheader(level.value)
        for tutorial in tutorials_by_level(level):
            _render_catalogue_entry(tutorial)


def _render_catalogue_entry(tutorial: Tutorial) -> None:
    pairs = f" · pairs with {', '.join(tutorial.pairs_with)}" if tutorial.pairs_with else ""
    with st.container(border=True):
        st.markdown(f"**{tutorial.label}** — {tutorial.track}{pairs}")
        st.caption(tutorial.summary)
        st.button(
            "Start" if tutorial.available else "Coming soon",
            key=f"tutorial_open_{tutorial.slug}",
            disabled=not tutorial.available,
            on_click=open_tutorial,
            args=(tutorial.slug,),
        )


def _render_tutorial(tutorial: Tutorial, jar: Path) -> None:
    steps = TUTORIAL_STEPS[tutorial.slug]
    step = st.session_state.get(STEP_KEY, 0)
    st.button("← All tutorials", on_click=open_tutorial, args=(None,))
    st.header(tutorial.label)
    st.progress((step + 1) / len(steps), text=f"Step {step + 1} of {len(steps)}")
    st.subheader(steps[step].title)
    steps[step].render(jar)
    previous_column, next_column = st.columns(2)
    previous_column.button("← Previous", disabled=step == 0, on_click=go_to_step, args=(step - 1,))
    next_column.button(
        "Next →", disabled=step == len(steps) - 1, on_click=go_to_step, args=(step + 1,)
    )


st.title("Tutorials")

selected_slug = st.session_state.get(SELECTED_KEY)
selected = next((t for t in TUTORIALS if t.slug == selected_slug and t.available), None)
if selected is None:
    # The catalogue itself does not need Evolver, only the tutorials' content does.
    render_sidebar()
    _render_catalogue()
else:
    _render_tutorial(selected, require_evolver_jar())
