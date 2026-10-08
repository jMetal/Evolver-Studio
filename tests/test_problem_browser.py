"""Tests for the helpers of the problem listing."""

from evolver_studio.problem_browser import add_problems


class TestAddProblems:
    def test_should_add_the_new_ones_after_the_chosen(self):
        # Act / Assert
        assert add_problems(["ZDT4"], ["DTLZ2", "WFG1"]) == ["ZDT4", "DTLZ2", "WFG1"]

    def test_should_leave_out_the_ones_already_chosen(self):
        # Act / Assert
        assert add_problems(["ZDT4", "DTLZ2"], ["DTLZ2", "WFG1", "WFG1"]) == [
            "ZDT4",
            "DTLZ2",
            "WFG1",
        ]
