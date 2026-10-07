"""The blinking label that shows a run (or a study) is in progress.

Streamlit has no such element, so it is a styled span.
"""

import streamlit as st

STYLE = """
<style>
@keyframes running-blink { 50% { opacity: 0.35; } }
.running-badge {
  display: inline-block;
  padding: 0.4rem 1rem;
  border-radius: 999px;
  border: 1px solid rgba(255, 75, 75, 0.7);
  background: rgba(255, 75, 75, 0.15);
  font-weight: 600;
  animation: running-blink 1s ease-in-out infinite;
}
</style>
"""


def inject_style() -> None:
    """Add the label's style to the page; once per page is enough."""
    st.markdown(STYLE, unsafe_allow_html=True)


def render(label: str) -> None:
    """Show the label.

    Args:
        label: What it says (the hourglass is added).
    """
    st.markdown(f'<span class="running-badge">⏳ {label}</span>', unsafe_allow_html=True)
