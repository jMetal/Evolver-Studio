"""The problems Evolver can resolve by name, as DescribeMain's `problemCatalogue` describes them.

Each problem has an encoding (an algorithm can only solve problems of an encoding it supports),
its number of objectives and variables when built with no arguments, and the arguments of its
constructor with arguments. Evolver's org.uma.evolver.cli.ProblemSpec takes those arguments all or
none, in order: a request names a problem as a plain string, or as a `{class, args}` map.

Evolver releases up to 2.3 have no `problemCatalogue`: `parse_problem_catalogue` then returns
None, and the pages fall back to the plain list of problem names.
"""

from collections.abc import Sequence
from dataclasses import dataclass

ArgumentValue = int | float | bool


@dataclass(slots=True, frozen=True)
class ProblemArgument:
    """An argument of a problem's constructor.

    Attributes:
        name: What it is (e.g. "numberOfObjectives").
        type: "integer", "number" (a decimal) or "boolean".
        default: Its value in the problem built with no arguments, or None when it is not known.
    """

    name: str
    type: str
    default: ArgumentValue | None


@dataclass(slots=True, frozen=True)
class Problem:
    """A problem Evolver can resolve by name.

    Attributes:
        name: The name a request uses (e.g. "DTLZ2").
        family: The family it belongs to (e.g. "DTLZ"); its problems share their arguments.
        encoding: The encoding of its solutions: "Double", "Binary" or "Permutation".
        number_of_objectives: Of the problem built with no arguments, or None if unknown.
        number_of_variables: Of the problem built with no arguments, or None if unknown.
        arguments: The arguments of its constructor with arguments, in order; empty for none.
    """

    name: str
    family: str
    encoding: str
    number_of_objectives: int | None
    number_of_variables: int | None
    arguments: tuple[ProblemArgument, ...]


def parse_problem_catalogue(manifest: dict) -> dict[str, Problem] | None:
    """Read the problem catalogue of DescribeMain's manifest.

    Args:
        manifest: The parsed manifest.

    Returns:
        The problems by name, or None when the manifest has no catalogue (Evolver 2.3 or older).
    """
    entries = manifest.get("problemCatalogue")
    if entries is None:
        return None
    problems = {}
    for entry in entries:
        arguments = tuple(
            ProblemArgument(argument["name"], argument["type"], argument.get("default"))
            for argument in entry.get("arguments") or ()
        )
        problems[entry["name"]] = Problem(
            name=entry["name"],
            family=entry["family"],
            encoding=entry["encoding"],
            number_of_objectives=entry.get("numberOfObjectives"),
            number_of_variables=entry.get("numberOfVariables"),
            arguments=arguments,
        )
    return problems


def problems_with_encoding(problems: dict[str, Problem], encoding: str) -> list[str]:
    """The names of the problems of an encoding, sorted.

    Args:
        problems: The problems by name.
        encoding: An encoding, e.g. "Double".

    Returns:
        Their names.
    """
    return sorted(name for name, problem in problems.items() if problem.encoding == encoding)


def problem_spec(name: str, arguments: Sequence[ArgumentValue | str] | None) -> str | dict:
    """The value of a request's `problem` (or of an entry of `trainingProblemNames`).

    Args:
        name: The problem's name.
        arguments: Its constructor's arguments, in order (a string for one of a class outside the
            catalogue, such as a file name), or None (or empty) to build it with none.

    Returns:
        The name alone, or a `{class, args}` map.
    """
    if not arguments:
        return name
    return {"class": name, "args": list(arguments)}


def problem_spec_name(spec: object) -> str | None:
    """The problem's name in a request's `problem` value.

    Args:
        spec: A plain name or a `{class, args}` map, as a request file holds it.

    Returns:
        The name, or None if the value is neither.
    """
    if isinstance(spec, str):
        return spec
    if isinstance(spec, dict) and isinstance(spec.get("class"), str):
        return spec["class"]
    return None


def problem_spec_arguments(spec: object) -> list[ArgumentValue]:
    """The constructor's arguments in a request's `problem` value.

    Args:
        spec: A plain name or a `{class, args}` map.

    Returns:
        The arguments, empty for a plain name.
    """
    if isinstance(spec, dict) and isinstance(spec.get("args"), list):
        return list(spec["args"])
    return []


def format_problem_spec(spec: object) -> str:
    """Describe a request's `problem` value in one line, e.g. "DTLZ2(12, 3)".

    Args:
        spec: A plain name or a `{class, args}` map.

    Returns:
        The name, followed by its arguments in parentheses when there are any.
    """
    name = problem_spec_name(spec) or str(spec)
    arguments = problem_spec_arguments(spec)
    if not arguments:
        return name
    return f"{name}({', '.join(_format_value(value) for value in arguments)})"


def parse_arguments_text(problem: Problem, text: str) -> list[ArgumentValue] | str:
    """Read a problem's arguments from a comma-separated text, e.g. "12, 3".

    Args:
        problem: The problem, whose arguments say how many values there are and their types.
        text: The text; blank for no arguments.

    Returns:
        The values, in order (empty for a blank text), or a message saying what is wrong.
    """
    if not text.strip():
        return []
    if not problem.arguments:
        return f"{problem.name} takes no arguments."
    pieces = [piece.strip() for piece in text.split(",")]
    expected = ", ".join(argument.name for argument in problem.arguments)
    if len(pieces) != len(problem.arguments):
        return (
            f"{problem.name} takes {len(problem.arguments)} arguments ({expected}), "
            f"or none; got {len(pieces)}."
        )
    values: list[ArgumentValue] = []
    for argument, piece in zip(problem.arguments, pieces, strict=True):
        value = _parse_value(argument.type, piece)
        if value is None:
            return f"{problem.name}: {argument.name} must be {_type_description(argument.type)}."
        values.append(value)
    return values


def format_arguments(problem: Problem) -> str:
    """Describe a problem's arguments and their defaults, e.g. "numberOfVariables=12, ...".

    Args:
        problem: The problem.

    Returns:
        One "name=default" per argument ("name" alone when its default is unknown), or "" for
        none.
    """
    return ", ".join(
        argument.name
        if argument.default is None
        else f"{argument.name}={_format_value(argument.default)}"
        for argument in problem.arguments
    )


def _parse_value(argument_type: str, text: str) -> ArgumentValue | None:
    if argument_type == "boolean":
        return {"true": True, "false": False}.get(text.lower())
    try:
        return int(text) if argument_type == "integer" else float(text)
    except ValueError:
        return None


def _type_description(argument_type: str) -> str:
    return {"integer": "an integer", "number": "a number", "boolean": "true or false"}.get(
        argument_type, argument_type
    )


def _format_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)
