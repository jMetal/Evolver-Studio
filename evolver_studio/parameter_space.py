"""Parsing and serialization of Evolver's YAML parameter space files.

Mirrors the schema org.uma.evolver.parameter.yaml.YAMLParameterSpace (and its
processors) actually read: a top-level mapping of parameter name to a
double/integer range, or a categorical parameter whose values can each unlock
their own nested conditional sub-parameters, plus optional global
sub-parameters always active regardless of the chosen value.

A categorical's `values` block round-trips in whichever form (a plain scalar
list, or a mapping keyed by choice name) the source file used: Evolver's own
parser treats a scalar list of numbers as an integer-valued category, but a
mapping's keys are always read back as strings — silently converting a
numeric list to mapping form would change that parameter's semantics.
"""

from dataclasses import dataclass, replace
from typing import Literal

import yaml

# YAMLParameterSpace's javadoc documents "int"/"integer" and "double"/"real" as accepted synonyms;
# normalized to the second spelling of each pair so RangeParameter.kind only ever holds one of two
# values. Confirmed against real files: MOPSO.yaml uses "real" throughout.
RANGE_TYPE_ALIASES = {"integer": "integer", "int": "integer", "double": "double", "real": "double"}


@dataclass(slots=True, frozen=True)
class RangeParameter:
    """A double or integer parameter with a numeric range.

    Attributes:
        name: The parameter's name.
        kind: "integer" or "double".
        lower_bound: The range's lower bound.
        upper_bound: The range's upper bound.
    """

    name: str
    kind: Literal["integer", "double"]
    lower_bound: float
    upper_bound: float


@dataclass(slots=True, frozen=True)
class CategoricalChoice:
    """One value a CategoricalParameter can take.

    Attributes:
        value: The choice's name (always a string — see module docstring).
        conditional_parameters: Sub-parameters only active when this choice
            is selected.
    """

    value: str
    conditional_parameters: tuple["ParameterSpec", ...] = ()


@dataclass(slots=True, frozen=True)
class CategoricalParameter:
    """A categorical parameter, whose values may unlock nested sub-parameters.

    Attributes:
        name: The parameter's name.
        choices: The values this parameter can take.
        global_sub_parameters: Sub-parameters always active, regardless of
            which choice is selected.
        uses_list_form: Whether `values` was written as a plain scalar list
            in the source YAML (vs. a mapping) — preserved for round-tripping.
        values_are_numeric: Whether the scalar list's values are numbers.
            Only meaningful when `uses_list_form` is True.
    """

    name: str
    choices: tuple[CategoricalChoice, ...]
    global_sub_parameters: tuple["ParameterSpec", ...] = ()
    uses_list_form: bool = False
    values_are_numeric: bool = False


ParameterSpec = RangeParameter | CategoricalParameter


def parse_parameter_space(text: str) -> list[ParameterSpec]:
    """Parse a parameter space YAML document's text.

    Args:
        text: The YAML document's content.

    Returns:
        The top-level parameters, in file order.

    Raises:
        ValueError: If a parameter's `type` is not recognized.
    """
    raw = yaml.safe_load(text) or {}
    return [_parse_parameter(name, spec) for name, spec in raw.items()]


def _parse_parameter(name: str, spec: dict) -> ParameterSpec:
    kind = spec["type"]
    normalized_kind = RANGE_TYPE_ALIASES.get(kind)
    if normalized_kind is not None:
        lower, upper = spec["range"]
        return RangeParameter(name=name, kind=normalized_kind, lower_bound=lower, upper_bound=upper)
    if kind == "categorical":
        return _parse_categorical(name, spec)
    raise ValueError(f"Unknown parameter type {kind!r} for parameter {name!r}")


def _parse_categorical(name: str, spec: dict) -> CategoricalParameter:
    values = spec["values"]
    global_sub_parameters = tuple(
        _parse_parameter(sub_name, sub_spec)
        for sub_name, sub_spec in (spec.get("globalSubParameters") or {}).items()
    )
    if isinstance(values, list):
        choices = tuple(CategoricalChoice(value=str(v)) for v in values)
        numeric = bool(values) and isinstance(values[0], (int, float))
        return CategoricalParameter(
            name, choices, global_sub_parameters, uses_list_form=True, values_are_numeric=numeric
        )
    choices = tuple(
        _parse_choice_entry(value_name, value_spec) for value_name, value_spec in values.items()
    )
    return CategoricalParameter(name, choices, global_sub_parameters)


def _parse_choice_entry(value_name: str, value_spec: dict | None) -> CategoricalChoice:
    conditional = (value_spec or {}).get("conditionalParameters") or {}
    sub_parameters = tuple(_parse_parameter(n, s) for n, s in conditional.items())
    return CategoricalChoice(value=value_name, conditional_parameters=sub_parameters)


def serialize_parameter_space(parameters: list[ParameterSpec]) -> str:
    """Serialize parameters back to the YAML format Evolver reads.

    Args:
        parameters: The top-level parameters to write, in the order given.

    Returns:
        The YAML document's text.
    """
    return yaml.safe_dump(
        {p.name: _to_yaml_node(p) for p in parameters}, sort_keys=False, default_flow_style=False
    )


def _to_yaml_node(parameter: ParameterSpec) -> dict:
    if isinstance(parameter, RangeParameter):
        return {"type": parameter.kind, "range": [parameter.lower_bound, parameter.upper_bound]}
    return _categorical_to_yaml_node(parameter)


def _categorical_to_yaml_node(parameter: CategoricalParameter) -> dict:
    node = {"type": "categorical", "values": _choices_to_yaml(parameter)}
    if parameter.global_sub_parameters:
        node["globalSubParameters"] = {
            sub.name: _to_yaml_node(sub) for sub in parameter.global_sub_parameters
        }
    return node


def _choices_to_yaml(parameter: CategoricalParameter):
    if parameter.uses_list_form:
        cast = _to_number if parameter.values_are_numeric else str
        return [cast(choice.value) for choice in parameter.choices]
    return {choice.value: _choice_value_node(choice) for choice in parameter.choices}


def _choice_value_node(choice: CategoricalChoice) -> dict:
    if not choice.conditional_parameters:
        return {}
    return {
        "conditionalParameters": {
            sub.name: _to_yaml_node(sub) for sub in choice.conditional_parameters
        }
    }


def _to_number(value: str) -> int | float:
    number = float(value)
    return int(number) if number.is_integer() else number


def with_selected_choices(
    parameter: CategoricalParameter, selected: set[str]
) -> CategoricalParameter:
    """Return a copy of `parameter` keeping only the given choices.

    Args:
        parameter: The categorical parameter to narrow.
        selected: The choice values (names) to keep.

    Returns:
        A copy with `choices` filtered to `selected`, preserving their order.
    """
    return replace(parameter, choices=tuple(c for c in parameter.choices if c.value in selected))


def with_range(parameter: RangeParameter, lower_bound: float, upper_bound: float) -> RangeParameter:
    """Return a copy of `parameter` with a narrowed range.

    Args:
        parameter: The range parameter to narrow.
        lower_bound: The new lower bound.
        upper_bound: The new upper bound.

    Returns:
        A copy with the updated bounds.
    """
    return replace(parameter, lower_bound=lower_bound, upper_bound=upper_bound)


def count_parameters(parameters: list[ParameterSpec]) -> int:
    """Count every parameter of a parameter space, sub-parameters included.

    The result matches Evolver's ParameterManagement.parameterFlattening: it is
    the number of variables of the flat encoding meta-optimizers search.

    Args:
        parameters: The top-level parameters.

    Returns:
        The total number of parameters, at any depth.
    """
    return sum(1 + count_parameters(_sub_parameters(parameter)) for parameter in parameters)


def _sub_parameters(parameter: ParameterSpec) -> list[ParameterSpec]:
    if isinstance(parameter, RangeParameter):
        return []
    conditional = [p for choice in parameter.choices for p in choice.conditional_parameters]
    return [*parameter.global_sub_parameters, *conditional]


def active_parameter_names(parameters: list[ParameterSpec], choices: dict[str, str]) -> list[str]:
    """List the parameters a configuration activates, in depth-first order.

    A parameter is active when its parent is active; a conditional parameter
    also needs its parent to take the value it hangs from. A categorical
    parameter missing from `choices` takes its first value.

    Args:
        parameters: The top-level parameters (always active).
        choices: The value chosen for each categorical parameter, by name.

    Returns:
        The names of the active parameters.
    """
    names = []
    for parameter in parameters:
        names.append(parameter.name)
        names.extend(active_parameter_names(active_sub_parameters(parameter, choices), choices))
    return names


def active_sub_parameters(parameter: ParameterSpec, choices: dict[str, str]) -> list[ParameterSpec]:
    """Return the sub-parameters of `parameter` that are active under `choices`.

    Args:
        parameter: An active parameter.
        choices: The value chosen for each categorical parameter, by name; a
            categorical parameter missing from it takes its first value.

    Returns:
        Its global sub-parameters followed by the conditional parameters of
        its chosen value; empty for a range parameter.
    """
    if isinstance(parameter, RangeParameter):
        return []
    chosen = choices.get(parameter.name, parameter.choices[0].value)
    conditional = [
        p
        for choice in parameter.choices
        if choice.value == chosen
        for p in choice.conditional_parameters
    ]
    return [*parameter.global_sub_parameters, *conditional]


@dataclass(slots=True, frozen=True)
class ParameterRow:
    """One parameter of a parameter space, flattened into a table row.

    Attributes:
        depth: Its nesting level (0 for a top-level parameter).
        name: The parameter's name.
        kind: "categorical", "integer" or "double".
        domain: Its range ("[min, max]") or its values, comma-separated.
        condition: When it applies: "parent = value" for a conditional
            parameter, "parent (any)" for a global sub-parameter, "" for a
            top-level one.
        parent: The index, in the same row list, of the parameter it hangs
            from; None for a top-level one.
    """

    depth: int
    name: str
    kind: str
    domain: str
    condition: str
    parent: int | None


def parameter_rows(parameters: list[ParameterSpec]) -> list[ParameterRow]:
    """Flatten a parameter space into table rows, depth-first and in file order.

    Each parameter is followed by its sub-parameters: the conditional ones of
    each value, in value order, and then the global ones.

    Args:
        parameters: The top-level parameters.

    Returns:
        One row per parameter, at any depth.
    """
    rows: list[ParameterRow] = []
    _append_rows(rows, parameters, 0, "", None)
    return rows


def _append_rows(
    rows: list[ParameterRow],
    parameters: list[ParameterSpec] | tuple[ParameterSpec, ...],
    depth: int,
    condition: str,
    parent: int | None,
) -> None:
    for parameter in parameters:
        index = len(rows)
        if isinstance(parameter, RangeParameter):
            domain = f"[{parameter.lower_bound}, {parameter.upper_bound}]"
            rows.append(
                ParameterRow(depth, parameter.name, parameter.kind, domain, condition, parent)
            )
            continue
        domain = ", ".join(choice.value for choice in parameter.choices)
        rows.append(ParameterRow(depth, parameter.name, "categorical", domain, condition, parent))
        for choice in parameter.choices:
            _append_rows(
                rows,
                choice.conditional_parameters,
                depth + 1,
                f"{parameter.name} = {choice.value}",
                index,
            )
        _append_rows(
            rows, parameter.global_sub_parameters, depth + 1, f"{parameter.name} (any)", index
        )


def filter_rows(rows: list[ParameterRow], text: str) -> list[ParameterRow]:
    """Keep the rows matching `text`, together with the rows they hang from.

    A row matches when its name, domain or condition contains `text`,
    ignoring case. Its ancestors are kept too, so a match never loses the
    context that explains when it applies.

    Args:
        rows: Rows built by `parameter_rows`.
        text: The text to look for; blank keeps every row.

    Returns:
        The kept rows, in their original order.
    """
    needle = text.strip().lower()
    if not needle:
        return list(rows)
    kept: set[int] = set()
    for index, row in enumerate(rows):
        if any(needle in field.lower() for field in (row.name, row.domain, row.condition)):
            current: int | None = index
            while current is not None and current not in kept:
                kept.add(current)
                current = rows[current].parent
    return [row for index, row in enumerate(rows) if index in kept]


@dataclass(slots=True, frozen=True)
class ParameterSpaceSummary:
    """Counts describing the size and shape of a parameter space.

    Attributes:
        total: Every parameter, at any depth.
        top_level: The top-level parameters.
        categorical: The categorical parameters, at any depth.
        numeric: The integer and double parameters, at any depth.
        max_depth: The deepest nesting level (0 when every parameter is
            top-level).
    """

    total: int
    top_level: int
    categorical: int
    numeric: int
    max_depth: int


def summarize_parameter_space(parameters: list[ParameterSpec]) -> ParameterSpaceSummary:
    """Count a parameter space's parameters by kind and depth.

    Args:
        parameters: The top-level parameters.

    Returns:
        Its summary; `total` equals `count_parameters(parameters)`.
    """
    rows = parameter_rows(parameters)
    categorical = sum(1 for row in rows if row.kind == "categorical")
    return ParameterSpaceSummary(
        total=len(rows),
        top_level=len(parameters),
        categorical=categorical,
        numeric=len(rows) - categorical,
        max_depth=max((row.depth for row in rows), default=0),
    )
