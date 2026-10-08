"""What the configurations of a training's final front have in common.

A training ends with a front of configurations, each a different compromise between its
meta-objectives. Where they agree on a parameter, the training found that value to matter; where
they differ, the parameter trades one objective for another or does not matter. Comparing them with
the algorithm's default configuration shows what the training changed about it.

Nothing here depends on Streamlit.
"""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import median

import pandas as pd

from evolver_studio.configuration import parameters_by_name, same_value
from evolver_studio.parameter_space import ParameterSpec, RangeParameter

# A numeric value is called the default's when it is this close to it, as a share of the range.
SAME_AS_DEFAULT_TOLERANCE = 0.05
SAME = "same"
DIFFERENT = "different"
UNKNOWN = "—"
COLUMNS = (
    "Parameter",
    "Kind",
    "Active in",
    "Typical value",
    "Agreement",
    "Values",
    "Default",
    "Vs default",
)


@dataclass(slots=True, frozen=True)
class ParameterSummary:
    """What the front's configurations do with one parameter.

    Attributes:
        name: The parameter.
        kind: "categorical" or "numeric".
        active: In how many configurations the parameter is active (they give it a value).
        total: How many configurations there are.
        typical: The most common value of a categorical parameter, the median of a numeric one.
        agreement: How much the configurations agree, from 0 to 1: the share that chooses the
            most common value of a categorical parameter, or one minus the spread of a numeric
            one (the interquartile range, as a share of its range in the parameter space; None
            when the space gives no range).
        values: The values the configurations give it: each choice with how many, or the range.
        default: The default configuration's value, or None when it is not known.
        versus_default: `SAME`, `DIFFERENT` or `UNKNOWN`.
    """

    name: str
    kind: str
    active: int
    total: int
    typical: str
    agreement: float | None
    values: str
    default: str | None
    versus_default: str


def summarize_front(
    configurations: Sequence[Mapping[str, str]],
    parameters: list[ParameterSpec],
    defaults: Mapping[str, str] | None = None,
) -> list[ParameterSummary]:
    """Summarize, parameter by parameter, what a front's configurations do.

    Args:
        configurations: The configurations of the front, each as its parameters' values by name
            (as `configuration.parse_configuration` reads them).
        parameters: The base algorithm's parameter space, which says what kind each parameter is
            and, for a numeric one, its range.
        defaults: The values of the algorithm's default configuration, or None if there is none.

    Returns:
        A summary of every parameter that some configuration gives a value, most agreed first.
    """
    by_name = parameters_by_name(parameters)
    names = list(dict.fromkeys(name for configuration in configurations for name in configuration))
    summaries = [
        _summarize(name, configurations, by_name.get(name), (defaults or {}).get(name))
        for name in names
    ]
    return sorted(
        summaries, key=lambda s: (-(s.agreement if s.agreement is not None else -1), s.name)
    )


def summaries_table(summaries: Sequence[ParameterSummary]) -> pd.DataFrame:
    """Lay the summaries out as a table.

    Args:
        summaries: What `summarize_front` returns.

    Returns:
        One row per parameter, with the columns `COLUMNS`.
    """
    rows = [
        {
            "Parameter": s.name,
            "Kind": s.kind,
            "Active in": f"{s.active} of {s.total}",
            "Typical value": s.typical,
            "Agreement": s.agreement,
            "Values": s.values,
            "Default": s.default if s.default is not None else UNKNOWN,
            "Vs default": s.versus_default,
        }
        for s in summaries
    ]
    return pd.DataFrame(rows, columns=list(COLUMNS))


def value_counts(configurations: Sequence[Mapping[str, str]], name: str) -> list[tuple[str, int]]:
    """How many configurations give each value to a parameter.

    Args:
        configurations: The configurations of the front.
        name: The parameter.

    Returns:
        The values and their counts, the most common first.
    """
    return _most_common_first(Counter(c[name] for c in configurations if name in c))


def numeric_values(configurations: Sequence[Mapping[str, str]], name: str) -> list[float]:
    """The values the configurations give to a numeric parameter.

    Args:
        configurations: The configurations of the front.
        name: The parameter.

    Returns:
        The values that are numbers, in the order of the configurations.
    """
    values = []
    for configuration in configurations:
        number = _number(configuration.get(name))
        if number is not None:
            values.append(number)
    return values


def _summarize(
    name: str,
    configurations: Sequence[Mapping[str, str]],
    parameter: ParameterSpec | None,
    default: str | None,
) -> ParameterSummary:
    given = [c[name] for c in configurations if name in c]
    numbers = [n for n in map(_number, given) if n is not None]
    # A parameter the space does not know is numeric when all its values are numbers.
    numeric = isinstance(parameter, RangeParameter) or (
        parameter is None and len(numbers) == len(given)
    )
    if numeric and numbers:
        return _summarize_numeric(name, given, numbers, parameter, default, len(configurations))
    return _summarize_categorical(name, given, default, len(configurations))


def _summarize_categorical(
    name: str, given: list[str], default: str | None, total: int
) -> ParameterSummary:
    counts = Counter(given)
    typical, share = max(counts.items(), key=lambda item: (item[1], item[0]))
    shown = ", ".join(f"{value} ×{count}" for value, count in _most_common_first(counts))
    if default is None:
        versus = UNKNOWN
    else:
        versus = SAME if all(same_value(value, default) for value in counts) else DIFFERENT
    return ParameterSummary(
        name, "categorical", len(given), total, typical, share / len(given), shown, default, versus
    )


def _most_common_first(counts: Counter) -> list[tuple[str, int]]:
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


def _summarize_numeric(
    name: str,
    given: list[str],
    numbers: list[float],
    parameter: ParameterSpec | None,
    default: str | None,
    total: int,
) -> ParameterSummary:
    ordered = sorted(numbers)
    typical = median(ordered)
    span = _range_span(parameter)
    quartiles = _quartiles(ordered)
    agreement = None if not span else max(1.0 - (quartiles[1] - quartiles[0]) / span, 0.0)
    default_number = _number(default)
    if default_number is None:
        versus = UNKNOWN
    elif span:
        versus = (
            SAME if abs(typical - default_number) <= SAME_AS_DEFAULT_TOLERANCE * span else DIFFERENT
        )
    else:
        versus = SAME if abs(typical - default_number) <= 1e-9 else DIFFERENT
    integer = isinstance(parameter, RangeParameter) and parameter.kind == "integer"
    return ParameterSummary(
        name=name,
        kind="numeric",
        active=len(given),
        total=total,
        typical=_format(typical, integer),
        agreement=agreement,
        values=f"{_format(ordered[0], integer)} – {_format(ordered[-1], integer)}",
        default=default,
        versus_default=versus,
    )


def _range_span(parameter: ParameterSpec | None) -> float | None:
    if isinstance(parameter, RangeParameter) and parameter.upper_bound > parameter.lower_bound:
        return float(parameter.upper_bound - parameter.lower_bound)
    return None


def _quartiles(ordered: list[float]) -> tuple[float, float]:
    if len(ordered) < 2:
        return ordered[0], ordered[0]
    series = pd.Series(ordered)
    return float(series.quantile(0.25)), float(series.quantile(0.75))


def _number(text: str | None) -> float | None:
    try:
        return float(text) if text is not None else None
    except ValueError:
        return None


def _format(value: float, integer: bool) -> str:
    return str(round(value)) if integer else f"{value:.4g}"
