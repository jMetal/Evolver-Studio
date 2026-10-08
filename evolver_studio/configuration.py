"""Configurations of a base-level algorithm: the `--parameter value ...` strings Evolver parses.

A configuration gives a value to every parameter of an algorithm's parameter space that is *active*
(see parameter_space.active_parameter_names). Evolver's default configurations
(defaultConfigurations/*.txt in its jar) and the `configuration` of a solve request use this format.
This module reads and writes it, and checks it against a parameter space; it has no Streamlit
dependency, so the guided editor of the Run algorithm page and its tests share it.
"""

import math
from collections.abc import Mapping

from evolver_studio.parameter_space import (
    CategoricalParameter,
    ParameterSpec,
    RangeParameter,
    active_parameter_names,
)


def parse_configuration(text: str) -> dict[str, str]:
    """Read a configuration string into a name-to-value mapping.

    Only the first non-blank line is read: a configuration file may list several, and Evolver's
    solve request uses the first.

    Args:
        text: A configuration, like "--crossover SBX --crossoverProbability 0.9".

    Returns:
        The values by parameter name, in the order they appear.

    Raises:
        ValueError: If the text is not a sequence of "--name value" pairs.
    """
    line = next((line for line in text.splitlines() if line.strip()), "")
    tokens = line.split()
    if len(tokens) % 2 != 0:
        raise ValueError(f"Expected '--name value' pairs, got an odd number of tokens: {line!r}")
    values: dict[str, str] = {}
    for name, value in zip(tokens[::2], tokens[1::2], strict=True):
        if not name.startswith("--"):
            raise ValueError(f"Expected a parameter name starting with '--', got {name!r}")
        values[name.removeprefix("--")] = value
    return values


def starting_values(parameters: list[ParameterSpec]) -> dict[str, str]:
    """Give every parameter of a space, at any depth, a value to start from.

    The first value of each categorical parameter and the middle of each range (rounded, for an
    integer one): what an algorithm without a default configuration starts from, and what fills the
    parameters a default configuration does not mention (those only a different choice activates).

    Args:
        parameters: The top-level parameters of a parameter space.

    Returns:
        One value per parameter name.
    """
    values: dict[str, str] = {}
    for parameter in parameters:
        if isinstance(parameter, RangeParameter):
            values[parameter.name] = _format_number(_middle(parameter), parameter)
            continue
        values[parameter.name] = parameter.choices[0].value
        children = [
            *parameter.global_sub_parameters,
            *(p for choice in parameter.choices for p in choice.conditional_parameters),
        ]
        values.update(starting_values(children))
    return values


def complete_values(parameters: list[ParameterSpec], values: Mapping[str, str]) -> dict[str, str]:
    """Fill a configuration with starting values for the parameters it does not give.

    Args:
        parameters: The top-level parameters of a parameter space.
        values: A configuration, possibly covering only some of the parameters.

    Returns:
        A value for every parameter of the space; the given ones kept as they are.
    """
    return {**starting_values(parameters), **values}


def configuration_string(parameters: list[ParameterSpec], values: Mapping[str, str]) -> str:
    """Write the configuration a set of values describes, with only the active parameters.

    Args:
        parameters: The top-level parameters of a parameter space.
        values: A value for every parameter, as `complete_values` returns.

    Returns:
        The "--name value ..." string, in the space's depth-first order.
    """
    names = active_parameter_names(parameters, dict(values))
    return " ".join(f"--{name} {values[name]}" for name in names)


def values_outside_the_space(
    parameters: list[ParameterSpec], values: Mapping[str, str]
) -> list[str]:
    """List the active parameters whose value the parameter space does not allow.

    Args:
        parameters: The top-level parameters of a parameter space.
        values: A value for every parameter, as `complete_values` returns.

    Returns:
        The names of the active parameters whose value is not one of a categorical parameter's
        values, or is not a number inside a range parameter's bounds.
    """
    by_name = parameters_by_name(parameters)
    return [
        name
        for name in active_parameter_names(parameters, dict(values))
        if not _allowed(by_name[name], values[name])
    ]


def modified_values(
    parameters: list[ParameterSpec], values: Mapping[str, str], reference: Mapping[str, str]
) -> list[str]:
    """List the active parameters whose value differs from a reference configuration.

    Numbers are compared as numbers ("0.9" equals "0.90"). A parameter the reference does not give
    counts as modified.

    Args:
        parameters: The top-level parameters of a parameter space.
        values: A value for every parameter, as `complete_values` returns.
        reference: The configuration to compare with, typically the default one.

    Returns:
        The names of the active parameters that differ, in the space's depth-first order.
    """
    return [
        name
        for name in active_parameter_names(parameters, dict(values))
        if name not in reference or not same_value(values[name], reference[name])
    ]


def parameters_by_name(parameters: list[ParameterSpec]) -> dict[str, ParameterSpec]:
    """Index a parameter space by parameter name, sub-parameters at any depth included.

    Args:
        parameters: The top-level parameters of a parameter space.

    Returns:
        Every parameter of the space, by name.
    """
    by_name: dict[str, ParameterSpec] = {}
    for parameter in parameters:
        by_name[parameter.name] = parameter
        if isinstance(parameter, CategoricalParameter):
            children = [
                *parameter.global_sub_parameters,
                *(p for choice in parameter.choices for p in choice.conditional_parameters),
            ]
            by_name.update(parameters_by_name(children))
    return by_name


def _allowed(parameter: ParameterSpec, value: str) -> bool:
    if isinstance(parameter, CategoricalParameter):
        return value in {choice.value for choice in parameter.choices}
    number = _as_number(value)
    return number is not None and parameter.lower_bound <= number <= parameter.upper_bound


def same_value(first: str, second: str) -> bool:
    """Tell whether two configuration values are the same: numbers as numbers ("0.9" is "0.90").

    Args:
        first: A value.
        second: Another value.

    Returns:
        Whether they are equal.
    """
    first_number, second_number = _as_number(first), _as_number(second)
    if first_number is not None and second_number is not None:
        return math.isclose(first_number, second_number)
    return first == second


def _as_number(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def _middle(parameter: RangeParameter) -> float:
    return (parameter.lower_bound + parameter.upper_bound) / 2


def _format_number(number: float, parameter: RangeParameter) -> str:
    return str(round(number)) if parameter.kind == "integer" else repr(float(number))
