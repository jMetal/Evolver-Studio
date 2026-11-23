"""Tests for the facts about what Evolver offers."""

from evolver_studio.app_state import Fact


class TestFactDetail:
    def test_should_list_the_examples_in_one_line(self):
        # Arrange
        fact = Fact(2, ("NSGA-II", "MOEAD"))

        # Act / Assert
        assert fact.detail() == "NSGA-II, MOEAD"

    def test_should_end_with_an_ellipsis_when_there_are_more_examples_than_shown(self):
        # Arrange
        fact = Fact(5, ("A", "B", "C", "D", "E"))

        # Act / Assert
        assert fact.detail(limit=3) == "A, B, C, …"

    def test_should_be_empty_without_examples(self):
        # Act / Assert
        assert Fact(3).detail() == ""
