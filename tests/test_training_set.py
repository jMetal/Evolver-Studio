"""Tests for training set table validation."""

from pathlib import Path

import pandas as pd
import pytest

from evolver_studio.problem_catalogue import Problem, ProblemArgument
from evolver_studio.training_set import (
    DEFAULT_EVALUATIONS,
    TRAINING_SET_COLUMNS,
    TrainingSet,
    parse_training_set,
    training_problem_specs,
    training_set_table,
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
    def test_should_start_with_no_arguments(self, tmp_path: Path):
        # Act
        training_set = parse_training_set(training_set_table(["ZDT4"], PROBLEMS, tmp_path))

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


class TestTrainingSetTable:
    @pytest.fixture
    def working_directory(self, tmp_path: Path) -> Path:
        fronts = tmp_path / "resources" / "referenceFronts"
        fronts.mkdir(parents=True)
        (fronts / "ZDT4.csv").write_text("0,1\n")
        (fronts / "DTLZ2.2D.csv").write_text("0,1\n")
        (fronts / "DTLZ2.3D.csv").write_text("0,0,1\n")
        return tmp_path

    def test_should_give_a_row_per_problem_with_its_front_and_the_default_budget(
        self, working_directory: Path
    ):
        # Act
        table = training_set_table(["ZDT4", "DTLZ2"], PROBLEMS, working_directory)

        # Assert: DTLZ2 has fronts for several objectives; the catalogue says it has 3
        assert list(table["problem"]) == ["ZDT4", "DTLZ2"]
        assert list(table["reference_front"]) == [
            "resources/referenceFronts/ZDT4.csv",
            "resources/referenceFronts/DTLZ2.3D.csv",
        ]
        assert list(table["evaluations"]) == [DEFAULT_EVALUATIONS, DEFAULT_EVALUATIONS]
        assert parse_training_set(table) is not None

    def test_should_keep_what_the_user_edited_of_a_problem_that_stays(
        self, working_directory: Path
    ):
        # Arrange
        edited = _table(
            {
                "problem": "DTLZ2",
                "arguments": "12, 2",
                "reference_front": "mine.csv",
                "evaluations": 500,
            }
        )

        # Act
        table = training_set_table(["ZDT4", "DTLZ2"], PROBLEMS, working_directory, edited)

        # Assert
        row = table.set_index("problem").loc["DTLZ2"]
        assert (row["arguments"], row["reference_front"], row["evaluations"]) == (
            "12, 2",
            "mine.csv",
            500,
        )
        assert list(table["problem"]) == ["ZDT4", "DTLZ2"]

    def test_should_accept_a_class_name_the_catalogue_does_not_know(self, working_directory):
        # Act
        table = training_set_table(["org.example.MyProblem"], {}, working_directory)

        # Assert: no front can be guessed for it, so the user fills it in
        assert list(table["problem"]) == ["org.example.MyProblem"]
        assert list(table["reference_front"]) == [""]

    def test_should_be_empty_without_problems(self, working_directory: Path):
        # Act
        table = training_set_table([], PROBLEMS, working_directory)

        # Assert
        assert table.empty
        assert list(table.columns) == list(TRAINING_SET_COLUMNS)
