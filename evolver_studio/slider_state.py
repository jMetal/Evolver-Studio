"""Pure decision logic for a slider that auto-tracks a growing bound until touched.

Deliberately Streamlit-agnostic (works on plain values, not st.session_state)
so it can be unit-tested without a UI.
"""


def next_slider_value(
    current_value: int | None, tracked_available: int | None, available: int
) -> tuple[int, int]:
    """Decide a slider's next value and what to record as auto-tracked.

    Passing a fresh `value=` to a Streamlit slider on every rerun re-applies
    it even after the user picks a smaller value (Streamlit re-applies
    `value` whenever `max_value` changes, not only on first creation). This
    computes what the slider's stored value should become instead: it keeps
    tracking `available` (e.g. "show everything") until the user moves it
    away from the last value that was auto-tracked, at which point their
    choice is left alone even as `available` keeps growing.

    Args:
        current_value: The slider's current stored value, or None if never set.
        tracked_available: The `available` count last auto-tracked, or None.
        available: The current number of distinct checkpoints available.

    Returns:
        (next_value, next_tracked_available).
    """
    untouched = current_value is None or current_value == tracked_available
    if untouched:
        return available, available
    return current_value, tracked_available
