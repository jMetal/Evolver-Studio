"""Tests for describing a run in progress."""

from evolver_studio.evolver_client import RunState, RunStatus
from evolver_studio.progress import running_label, study_running_label


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
