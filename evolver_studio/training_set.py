"""A training set: parallel lists of training problems, their reference fronts, and evaluations.

Mirrors org.uma.evolver.cli.training.BaseLevelConfig's three parallel fields
(trainingProblemNames/trainingReferenceFrontFileNames/trainingEvaluations) —
the CLI does not resolve training sets by name (see Evolver's
cli-training-prototype.md), so a training set is always spelled out
explicitly, one row per problem.
"""

from dataclasses import dataclass

import pandas as pd

TRAINING_SET_COLUMNS = ("problem", "reference_front", "evaluations")


def default_training_set_table() -> pd.DataFrame:
    """A single-row starting table (ZDT4), editable from there.

    Returns:
        A DataFrame with TRAINING_SET_COLUMNS, one row.
    """
    row = {
        "problem": "ZDT4",
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
    """

    problem_names: list[str]
    reference_front_file_names: list[str]
    evaluations: list[int]


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
    return TrainingSet(problem_names, reference_front_file_names, evaluations)
