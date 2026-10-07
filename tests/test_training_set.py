"""Tests for training set table validation."""

import pandas as pd
import pytest

from evolver_studio.problem_catalogue import Problem, ProblemArgument
from evolver_studio.training_set import (
    TRAINING_SET_COLUMNS,
    TrainingSet,
    default_training_set_table,
    parse_training_set,
    training_problem_specs,
)

PROBLEMS = {
    "DTLZ2": Problem(
        "DTLZ2",
        "DTLZ",
        "Double",
        3,
        12,
        (
            ProblemArgument("numberOfVariables", "integer", 12),
            ProblemArgument("numberOfObjectives", "integer", 3),
        ),
    ),
    "ZDT5": Problem(
        "ZDT5", "ZDT", "Binary", 2, 11, (ProblemArgument("numberOfVariables", "integer", 11),)
    ),
    "ZDT4": Problem("ZDT4", "ZDT", "Double", 2, 10, ()),
}


def _table(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(list(rows), columns=list(TRAINING_SET_COLUMNS))


class TestParseTrainingSet:
    def test_should_return_none_for_an_empty_table(self):
        """No rows means no training set to launch with."""
        # Act / Assert
        assert parse_training_set(_table()) is None

    def test_should_parse_a_single_row(self):
        # Arrange
        table = _table({"problem": "ZDT4", "reference_front": "front.csv", "evaluations": 12000})

        # Act
        training_set = parse_training_set(table)

        # Assert
        assert training_set.problem_names == ["ZDT4"]
        assert training_set.reference_front_file_names == ["front.csv"]
        assert training_set.evaluations == [12000]

    def test_should_parse_multiple_rows_in_order(self):
        # Arrange
        table = _table(
            {"problem": "RE31", "reference_front": "RE31.csv", "evaluations": 10000},
            {"problem": "RE32", "reference_front": "RE32.csv", "evaluations": 20000},
        )

        # Act
        training_set = parse_training_set(table)

        # Assert
        assert training_set.problem_names == ["RE31", "RE32"]
        assert training_set.reference_front_file_names == ["RE31.csv", "RE32.csv"]
        assert training_set.evaluations == [10000, 20000]

    def test_should_reject_a_row_with_a_blank_problem(self):
        # Arrange
        table = _table({"problem": "", "reference_front": "front.csv", "evaluations": 12000})

        # Act / Assert
        assert parse_training_set(table) is None

    def test_should_reject_a_row_with_a_blank_reference_front(self):
        # Arrange
        table = _table({"problem": "ZDT4", "reference_front": "", "evaluations": 12000})

        # Act / Assert
        assert parse_training_set(table) is None

    def test_should_reject_a_row_with_missing_evaluations(self):
        # Arrange
        table = _table({"problem": "ZDT4", "reference_front": "front.csv", "evaluations": None})

        # Act / Assert
        assert parse_training_set(table) is None

    def test_should_reject_a_row_with_non_positive_evaluations(self):
        # Arrange
        table = _table({"problem": "ZDT4", "reference_front": "front.csv", "evaluations": 0})

        # Act / Assert
        assert parse_training_set(table) is None

    def test_should_cast_evaluations_to_int(self):
        """The data editor's number column may hand back a float."""
        # Arrange
        table = _table({"problem": "ZDT4", "reference_front": "front.csv", "evaluations": 12000.0})

        # Act
        training_set = parse_training_set(table)

        # Assert
        assert training_set.evaluations == [12000]
        assert isinstance(training_set.evaluations[0], int)


class TestArgumentsColumn:
    def test_should_start_with_no_arguments(self):
        # Act
        training_set = parse_training_set(default_training_set_table())

        # Assert
        assert training_set is not None
        assert training_set.arguments_texts == [""]

    def test_should_keep_the_arguments_as_typed(self):
        # Arrange
        table = _table(
            {"problem": "DTLZ2", "arguments": "12, 2", "reference_front": "f", "evaluations": 1},
            {"problem": "ZDT4", "arguments": None, "reference_front": "f", "evaluations": 1},
        )

        # Act
        training_set = parse_training_set(table)

        # Assert
        assert training_set is not None
        assert training_set.arguments_texts == ["12, 2", ""]


def _training_set(*rows: tuple[str, str]) -> TrainingSet:
    names = [name for name, _ in rows]
    return TrainingSet(names, ["f"] * len(rows), [1000] * len(rows), [text for _, text in rows])


class TestTrainingProblemSpecs:
    def test_should_build_a_class_and_args_entry_for_a_problem_with_arguments(self):
        # Act
        specs, errors = training_problem_specs(
            _training_set(("DTLZ2", "12, 2"), ("ZDT4", "")), PROBLEMS, "Double"
        )

        # Assert
        assert errors == []
        assert specs == [{"class": "DTLZ2", "args": [12, 2]}, "ZDT4"]

    def test_should_reject_a_problem_of_another_encoding(self):
        # Act
        _, errors = training_problem_specs(_training_set(("ZDT5", "")), PROBLEMS, "Double")

        # Assert
        assert errors == [
            "ZDT5 is a Binary problem, but the base algorithm is run with the Double encoding."
        ]

    def test_should_reject_arguments_that_do_not_fit_the_problem(self):
        # Act
        _, errors = training_problem_specs(_training_set(("DTLZ2", "12")), PROBLEMS, "Double")

        # Assert
        assert len(errors) == 1 and "takes 2 arguments" in errors[0]

    @pytest.mark.parametrize("problems", [None, {}], ids=["no catalogue", "not in it"])
    def test_should_pass_on_a_problem_it_does_not_know_with_its_arguments(self, problems):
        # Arrange
        name = "org.uma.jmetal.problem.multiobjective.wfg.WFG1"

        # Act
        specs, errors = training_problem_specs(
            _training_set((name, "2, 4, true, 0.5, data.txt")), problems, "Double"
        )

        # Assert
        assert errors == []
        assert specs == [{"class": name, "args": [2, 4, True, 0.5, "data.txt"]}]

    def test_should_accept_a_training_set_without_arguments_texts(self):
        # Act
        specs, errors = training_problem_specs(
            TrainingSet(["ZDT4"], ["f"], [1000]), PROBLEMS, "Double"
        )

        # Assert
        assert (specs, errors) == (["ZDT4"], [])
