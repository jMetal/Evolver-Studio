"""Streamlit form rendering for editing a parsed parameter space.

Streamlit-dependent (widgets), kept separate from parameter_space.py so the
parsing/serialization/editing logic stays independently testable.
"""

from dataclasses import replace

import streamlit as st

from evolver_studio.parameter_space import (
    CategoricalChoice,
    CategoricalParameter,
    ParameterSpec,
    RangeParameter,
    with_range,
    with_selected_choices,
)


def render_parameter_form(parameters: list[ParameterSpec], key_prefix: str) -> list[ParameterSpec]:
    """Render an editable form for a parameter space, returning the edited tree.

    Args:
        parameters: The top-level parameters to render.
        key_prefix: Prefix for widget keys, unique per form instance.

    Returns:
        A new parameter list reflecting the current widget values.
    """
    return [_render_parameter(p, f"{key_prefix}_{p.name}") for p in parameters]


def _render_parameter(parameter: ParameterSpec, key: str) -> ParameterSpec:
    if isinstance(parameter, RangeParameter):
        return _render_range(parameter, key)
    return _render_categorical(parameter, key)


def _render_range(parameter: RangeParameter, key: str) -> RangeParameter:
    lower, upper = st.slider(
        parameter.name,
        min_value=parameter.lower_bound,
        max_value=parameter.upper_bound,
        value=(parameter.lower_bound, parameter.upper_bound),
        key=key,
    )
    return with_range(parameter, lower, upper)


def _selected_choice_values(parameter: CategoricalParameter, key: str) -> list[str]:
    """Let the user narrow which choices remain active, never leaving none selected."""
    all_values = [choice.value for choice in parameter.choices]
    selected = st.multiselect(parameter.name, options=all_values, default=all_values, key=key)
    if not selected:
        st.warning(f'"{parameter.name}" needs at least one value selected; keeping all.')
        return all_values
    return selected


def _render_choice(choice: CategoricalChoice, key: str) -> CategoricalChoice:
    if not choice.conditional_parameters:
        return choice
    with st.expander(choice.value):
        rendered = tuple(
            _render_parameter(sub, f"{key}_{sub.name}") for sub in choice.conditional_parameters
        )
    return replace(choice, conditional_parameters=rendered)


def _render_categorical(parameter: CategoricalParameter, key: str) -> CategoricalParameter:
    selected_values = _selected_choice_values(parameter, f"{key}_choices")
    narrowed = with_selected_choices(parameter, set(selected_values))
    choices = tuple(_render_choice(choice, f"{key}_{choice.value}") for choice in narrowed.choices)
    global_sub_parameters = tuple(
        _render_parameter(sub, f"{key}_{sub.name}") for sub in parameter.global_sub_parameters
    )
    return replace(narrowed, choices=choices, global_sub_parameters=global_sub_parameters)
