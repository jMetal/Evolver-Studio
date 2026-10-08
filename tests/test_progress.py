"""Tests for describing a run in progress."""

from evolver_studio.evolver_client import RunState, RunStatus
from evolver_studio.progress import running_label, study_running_label, training_progress


def _status(done: int, total: int) -> RunStatus:
    return RunStatus(RunState.RUNNING, done, total, "2026-10-07T12:00:00")


class TestRunningLabel:
    def test_should_say_only_running_before_the_first_status(self):
        # Act / Assert
        assert running_label(None, 1) == "Running…"
        assert running_label(_status(0, 25000), 1) == "Running…"

    def test_should_give_the_evaluation_of_a_single_run(self):
        # Act / Assert
        assert running_label(_status(12000, 25000), 1) == "Running · evaluation 12,000 of 25,000"

    def test_should_say_which_run_is_in_progress_with_several(self):
        """Evolver counts the evaluations over all the runs: 3 runs of 1000 make 3000."""
        # Act
        label = running_label(_status(1500, 3000), 3)

        # Assert
        assert label == "Running · evaluation 1,500 of 3,000 · run 2 of 3"

    def test_should_not_go_past_the_last_run_when_all_are_done(self):
        # Act / Assert
        assert running_label(_status(3000, 3000), 3).endswith("run 3 of 3")


class TestStudyRunningLabel:
    def test_should_count_the_jobs_finished(self):
        # Act / Assert
        assert study_running_label(_status(3, 12)) == "Running · 3 of 12 jobs finished"

    def test_should_say_only_running_before_the_first_status(self):
        # Act / Assert
        assert study_running_label(None) == "Running…"


class TestTrainingProgress:
    def test_should_give_the_evaluations_done_of_those_to_do(self):
        # Act
        fraction, text = training_progress(_status(500, 2000))

        # Assert
        assert (fraction, text) == (0.25, "500/2000")

    def test_should_not_go_past_one(self):
        # Act / Assert
        assert training_progress(_status(2100, 2000))[0] == 1.0

    def test_should_give_the_time_spent_of_the_limit_for_a_run_limited_by_time(self):
        """Such a run has no evaluations to do: its status says 0, and the fraction is of time."""
        # Arrange
        status = RunStatus(
            RunState.RUNNING, 340, 0, "x", max_computing_time_minutes=10.0, elapsed_minutes=2.5
        )

        # Act
        fraction, text = training_progress(status)

        # Assert
        assert fraction == 0.25
        assert text == "2.5 of 10 min · 340 evaluations"

    def test_should_say_zero_time_before_the_first_status_of_a_timed_run(self):
        # Arrange
        status = RunStatus(RunState.RUNNING, 0, 0, "x", max_computing_time_minutes=2.0)

        # Act / Assert
        assert training_progress(status)[0] == 0.0
