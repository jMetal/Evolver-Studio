"""Which tutorial is open in the Tutorials page, and which of its steps.

The page keeps it in the session state. It is shared with the steps of the tutorials, because a
link to the Tutorials page does nothing from inside it (it is the page already shown, with a
tutorial still selected): leaving a tutorial means clearing that selection, with a button.
"""

import streamlit as st

SELECTED_KEY = "tutorial_selected"
STEP_KEY = "tutorial_step"


def open_tutorial(tutorial_id: str | None) -> None:
    """Open a tutorial at its first step, or the catalogue when None (a button's callback).

    Args:
        tutorial_id: The tutorial's id (e.g. "S2"), or None for the list of tutorials.
    """
    st.session_state[SELECTED_KEY] = tutorial_id
    st.session_state[STEP_KEY] = 0


def go_to_step(step: int) -> None:
    """Show a step of the tutorial open (a button's callback).

    Args:
        step: The step's index, from 0.
    """
    st.session_state[STEP_KEY] = step
