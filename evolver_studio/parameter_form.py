"""Streamlit rendering of a parsed parameter space: an editable form, and read-only views.

Streamlit-dependent (widgets), kept separate from parameter_space.py so the
parsing/serialization/editing logic stays independently testable.
"""

import math
from collections.abc import Mapping
from dataclasses import replace

import streamlit as st

from evolver_studio.configuration import same_value
from evolver_studio.parameter_space import (
    CategoricalChoice,
    CategoricalParameter,
    ParameterRow,
    ParameterSpec,
    RangeParameter,
    active_sub_parameters,
    filter_rows,
    parameter_rows,
    summarize_parameter_space,
    with_range,
    with_selected_choices,
)

# Rows shown before the table scrolls: enough for most spaces, short enough to keep the rest of the
# page in sight.
TABLE_HEIGHT_IN_ROWS = 15
# st.dataframe's default row height, in pixels, plus the header row.
_ROW_HEIGHT = 35


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
    docstring, editing lives in the Training page's form instead.

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
            lines.append(f"{indent}    - *if {choice.value}:*")
            lines.extend(_readonly_lines(list(choice.conditional_parameters), level + 2))
    if parameter.global_sub_parameters:
        lines.append(f"{indent}    - *always:*")
        lines.extend(_readonly_lines(list(parameter.global_sub_parameters), level + 2))
    return lines


def render_parameter_space_table(parameters: list[ParameterSpec], key: str) -> None:
    """Render a parameter space as a read-only table, one row per parameter.

    Above the table, a one-line summary of the space's size and a filter by text. The
    hierarchy shows in two ways: each sub-parameter's name is indented under
    the parameter it hangs from, and its "Active if" column says when it
    applies. Filtering keeps the ancestors of every match.

    Args:
        parameters: The top-level parameters to render.
        key: Prefix for widget keys, unique per table on the page.
    """
    summary = summarize_parameter_space(parameters)
    # One line, not st.metric tiles: the point of the table is to keep the space in a single view.
    st.markdown(
        f"**{summary.total}** parameters · **{summary.top_level}** top-level · "
        f"**{summary.categorical}** categorical · **{summary.numeric}** numeric · "
        f"maximum depth **{summary.max_depth}**"
    )
    text = st.text_input(
        "Filter",
        key=f"{key}_filter",
        placeholder="Name, value or condition (e.g. SBX, archive, mutation)",
    )
    rows = filter_rows(parameter_rows(parameters), text)
    if not rows:
        st.info(f'No parameter matches "{text}".')
        return
    st.dataframe(
        [_table_row(row) for row in rows],
        hide_index=True,
        width="stretch",
        height=_ROW_HEIGHT * (min(len(rows), TABLE_HEIGHT_IN_ROWS) + 1) + 3,
        # Domain last: a long list of values can then overflow without hiding another column.
        column_config={
            "Parameter": st.column_config.TextColumn("Parameter", width=300),
            "Type": st.column_config.TextColumn("Type", width=100),
            "Active if": st.column_config.TextColumn("Active if", width=280),
            "Domain": st.column_config.TextColumn("Domain", width="large"),
        },
        key=f"{key}_table",
    )


def _table_row(row: ParameterRow) -> dict[str, str]:
    # Non-breaking spaces: the table would trim ordinary leading ones.
    indent = "\u00a0" * 5 * (row.depth - 1) + "\u00a0└\u00a0" if row.depth else ""
    return {
        "Parameter": f"{indent}{row.name}",
        "Type": row.kind,
        "Active if": row.condition,
        "Domain": row.domain,
    }


CONFIGURATION_FORM_COLUMNS = 3
MODIFIED_MARK = " ✏️"


def render_configuration_form(
    parameters: list[ParameterSpec],
    values: Mapping[str, str],
    reference: Mapping[str, str],
    key_prefix: str,
) -> dict[str, str]:
    """Render a form to adjust a configuration within a parameter space.

    One widget per *active* parameter, filled with the configuration's value and limited to what
    the space allows: a selector with a categorical parameter's values, a number input with a range
    parameter's bounds. A parameter's sub-parameters appear and disappear with its value, and a
    parameter whose value differs from the reference configuration is marked. A value the space
    does not allow is shown as it is, flagged, not corrected in silence.

    Args:
        parameters: The top-level parameters of the space.
        values: A value for every parameter of the space, as `configuration.complete_values` gives.
        reference: The configuration to mark changes against, typically the default one.
        key_prefix: Prefix for widget keys, unique per form instance; a new prefix resets the form
            to `values`.

    Returns:
        The values after the user's edits: `values`, with the rendered parameters' values replaced.
    """
    edited = dict(values)
    for group in _configuration_groups(parameters, values, key_prefix):
        with st.container(border=True):
            columns = st.columns(CONFIGURATION_FORM_COLUMNS)
            position = [0]
            for parameter in group:
                _render_configuration_parameter(
                    parameter, edited, reference, key_prefix, columns, position
                )
    return edited


def _configuration_groups(
    parameters: list[ParameterSpec], values: Mapping[str, str], key_prefix: str
) -> list[list[ParameterSpec]]:
    """Group the top-level parameters into boxes: one for each with active sub-parameters, and
    one for each run of consecutive parameters without, which share a box instead of taking a
    box each."""
    groups: list[list[ParameterSpec]] = []
    previous_was_simple = False
    for parameter in parameters:
        current = st.session_state.get(f"{key_prefix}_{parameter.name}", values[parameter.name])
        simple = not isinstance(parameter, CategoricalParameter) or not active_sub_parameters(
            parameter, {parameter.name: str(current)}
        )
        if simple and previous_was_simple:
            groups[-1].append(parameter)
        else:
            groups.append([parameter])
        previous_was_simple = simple
    return groups


def _render_configuration_parameter(
    parameter: ParameterSpec,
    edited: dict[str, str],
    reference: Mapping[str, str],
    key_prefix: str,
    columns: list,
    position: list[int],
) -> None:
    """Render one parameter's widget in the next free column, then its active sub-parameters."""
    column = columns[position[0] % len(columns)]
    position[0] += 1
    key = f"{key_prefix}_{parameter.name}"
    current = st.session_state.get(key, edited[parameter.name])
    modified = parameter.name not in reference or not same_value(
        str(current), reference[parameter.name]
    )
    label = parameter.name + (MODIFIED_MARK if modified else "")
    with column:
        if isinstance(parameter, RangeParameter):
            edited[parameter.name] = _range_widget(parameter, label, edited[parameter.name], key)
            return
        edited[parameter.name] = _choice_widget(parameter, label, edited[parameter.name], key)
    for child in active_sub_parameters(parameter, {parameter.name: edited[parameter.name]}):
        _render_configuration_parameter(child, edited, reference, key_prefix, columns, position)


def _choice_widget(parameter: CategoricalParameter, label: str, value: str, key: str) -> str:
    allowed = [choice.value for choice in parameter.choices]
    options = allowed if value in allowed else [*allowed, value]
    return st.selectbox(
        label,
        options,
        index=options.index(value),
        format_func=lambda option: (
            option if option in allowed else f"{option} (not in the parameter space)"
        ),
        key=key,
    )


def _range_widget(parameter: RangeParameter, label: str, value: str, key: str) -> str:
    try:
        number = float(value)
    except ValueError:
        number = parameter.lower_bound
    lower, upper = min(parameter.lower_bound, number), max(parameter.upper_bound, number)
    help_text = f"{parameter.kind}, from {parameter.lower_bound:g} to {parameter.upper_bound:g}"
    if not parameter.lower_bound <= number <= parameter.upper_bound:
        help_text = f"{value} is outside the parameter space ({help_text})"
    if parameter.kind == "integer":
        chosen = st.number_input(
            label,
            min_value=int(lower),
            max_value=int(upper),
            value=int(number),
            step=1,
            help=help_text,
            key=key,
        )
        return str(int(chosen))
    chosen = st.number_input(
        label,
        min_value=float(lower),
        max_value=float(upper),
        value=float(number),
        step=_step(parameter),
        format="%.6g",
        help=help_text,
        key=key,
    )
    return repr(float(f"{chosen:.10g}"))


def _step(parameter: RangeParameter) -> float:
    """A round step of about a hundredth of the range."""
    return float(
        10 ** math.floor(math.log10((parameter.upper_bound - parameter.lower_bound) / 100))
    )
