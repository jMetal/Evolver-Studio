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


def render_parameter_space_readonly(parameters: list[ParameterSpec]) -> None:
    """Render a parameter space as a compact, read-only nested list.

    No widgets: a single markdown tree, for browsing a parameter space
    without implying it can be edited here — see parameter_form.py's module
    docstring, editing lives in the Entrenamiento page's form instead.

    Args:
        parameters: The top-level parameters to render.
    """
    st.markdown("\n".join(_readonly_lines(parameters, 0)))


def _readonly_lines(parameters: list[ParameterSpec], level: int) -> list[str]:
    lines = []
    for parameter in parameters:
        if isinstance(parameter, RangeParameter):
            lines.append(_readonly_range_line(parameter, level))
        else:
            lines.extend(_readonly_categorical_lines(parameter, level))
    return lines


def _readonly_range_line(parameter: RangeParameter, level: int) -> str:
    indent = "    " * level
    bounds = f"[{parameter.lower_bound}, {parameter.upper_bound}]"
    return f"{indent}- **{parameter.name}**: {bounds} ({parameter.kind})"


def _readonly_categorical_lines(parameter: CategoricalParameter, level: int) -> list[str]:
    indent = "    " * level
    values = ", ".join(choice.value for choice in parameter.choices)
    lines = [f"{indent}- **{parameter.name}**: {values}"]
    for choice in parameter.choices:
        if choice.conditional_parameters:
            lines.append(f"{indent}    - *si {choice.value}:*")
            lines.extend(_readonly_lines(list(choice.conditional_parameters), level + 2))
    if parameter.global_sub_parameters:
        lines.append(f"{indent}    - *siempre:*")
        lines.extend(_readonly_lines(list(parameter.global_sub_parameters), level + 2))
    return lines
