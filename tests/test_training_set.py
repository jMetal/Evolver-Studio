"""Tests for training set table validation."""

import pandas as pd

from evolver_studio.training_set import TRAINING_SET_COLUMNS, parse_training_set


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
