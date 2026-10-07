"""A training set: parallel lists of training problems, their reference fronts, and evaluations.

Mirrors org.uma.evolver.cli.training.BaseLevelConfig's three parallel fields
(trainingProblemNames/trainingReferenceFrontFileNames/trainingEvaluations) —
the CLI does not resolve training sets by name (see Evolver's
cli-training-prototype.md), so a training set is always spelled out
explicitly, one row per problem. A problem may be given its constructor's arguments, as a
comma-separated text, which makes it a `{class, args}` entry of trainingProblemNames.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field

import pandas as pd

from evolver_studio.problem_catalogue import (
    ArgumentValue,
    Problem,
    parse_arguments_text,
    problem_spec,
)

TRAINING_SET_COLUMNS = ("problem", "arguments", "reference_front", "evaluations")


def default_training_set_table() -> pd.DataFrame:
    """A single-row starting table (ZDT4), editable from there.

    Returns:
        A DataFrame with TRAINING_SET_COLUMNS, one row.
    """
    row = {
        "problem": "ZDT4",
        "arguments": "",
        "reference_front": "resources/referenceFronts/ZDT4.csv",
        "evaluations": 12000,
    }
    return pd.DataFrame([row], columns=list(TRAINING_SET_COLUMNS))


@dataclass(slots=True, frozen=True)
class TrainingSet:
    """A validated training set, ready to become BaseLevelConfig's parallel lists.

    Attributes:
        problem_names: Training problem names, one per row.
        reference_front_file_names: Reference front file path, one per problem.
        evaluations: Base-level evaluation budget, one per problem.
        arguments_texts: The arguments given to each problem, as typed ("" for none), one per
            problem; empty when the table has no arguments column.
    """

    problem_names: list[str]
    reference_front_file_names: list[str]
    evaluations: list[int]
    arguments_texts: list[str] = field(default_factory=list)


def parse_training_set(table: pd.DataFrame) -> TrainingSet | None:
    """Validate and convert an edited training-set table.

    Args:
        table: The edited table, expected to have TRAINING_SET_COLUMNS.

    Returns:
        The parsed TrainingSet, or None if there are no rows or any row is
        incomplete (blank problem/reference front, missing/non-positive
        evaluations).
    """
    if table.empty:
        return None
    problem_names: list[str] = []
    reference_front_file_names: list[str] = []
    evaluations: list[int] = []
    arguments_texts: list[str] = []
    for _, row in table.iterrows():
        problem = row.get("problem")
        reference_front = row.get("reference_front")
        row_evaluations = row.get("evaluations")
        if (
            not problem
            or not reference_front
            or row_evaluations is None
            or pd.isna(row_evaluations)
            or row_evaluations <= 0
        ):
            return None
        problem_names.append(str(problem))
        reference_front_file_names.append(str(reference_front))
        evaluations.append(int(row_evaluations))
        arguments = row.get("arguments")
        arguments_texts.append("" if arguments is None or pd.isna(arguments) else str(arguments))
    return TrainingSet(problem_names, reference_front_file_names, evaluations, arguments_texts)


def training_problem_specs(
    training_set: TrainingSet, problems: Mapping[str, Problem] | None, encoding: str
) -> tuple[list[str | dict], list[str]]:
    """Turn a training set's problems into trainingProblemNames' entries, checking them.

    A problem of the catalogue must have the base algorithm's encoding, and its arguments must be
    all of its constructor's or none. A problem the catalogue does not describe (a fully-qualified
    class name, or any problem with a jar that has no catalogue) is passed on as it is, its
    arguments read as integers, numbers or booleans: Evolver matches them against a constructor.

    Args:
        training_set: The parsed training set.
        problems: The problem catalogue, or None when the jar has none.
        encoding: The encoding the base algorithm is run with.

    Returns:
        The entries, one per problem, and the messages of what is wrong (empty when nothing is).
    """
    specs: list[str | dict] = []
    errors: list[str] = []
    texts = training_set.arguments_texts or [""] * len(training_set.problem_names)
    for name, text in zip(training_set.problem_names, texts, strict=True):
        described = problems.get(name) if problems is not None else None
        arguments: list[ArgumentValue | str]
        if described is None:
            arguments = _parse_untyped_arguments(text)
        else:
            if described.encoding != encoding:
                errors.append(
                    f"{name} is a {described.encoding} problem, but the base algorithm is run "
                    f"with the {encoding} encoding."
                )
            parsed = parse_arguments_text(described, text)
            if isinstance(parsed, str):
                errors.append(parsed)
                arguments = []
            else:
                arguments = list(parsed)
        specs.append(problem_spec(name, arguments))
    return specs, errors


def _parse_untyped_arguments(text: str) -> list[ArgumentValue | str]:
    """Read comma-separated arguments of unknown types: integers, numbers, booleans or strings."""
    values: list[ArgumentValue | str] = []
    for piece in (piece.strip() for piece in text.split(",")):
        if not piece:
            continue
        if piece.lower() in ("true", "false"):
            values.append(piece.lower() == "true")
            continue
        try:
            values.append(int(piece))
        except ValueError:
            try:
                values.append(float(piece))
            except ValueError:
                values.append(piece)
    return values
