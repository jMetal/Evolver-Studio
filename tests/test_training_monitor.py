"""Tests for following a training run through the files Evolver appends to."""

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from evolver_studio.evolver_client import RunState, RunStatus
from evolver_studio.training_monitor import (
    CheckpointReader,
    CheckpointSummary,
    VarConfFollower,
    build_convergence_figure,
    build_population_figure,
    format_duration,
    summarize_run,
)

HEADER = "Evaluation,SolutionId,EP,NHV\n"
BLOCK_100 = "100,0,5.0,0.9\n100,1,4.0,0.8\n100,2,3.0,0.7\n"
BLOCK_200 = "200,0,2.0,0.5\n200,1,1.0,0.4\n"
BLOCK_300 = "300,0,0.5,0.2\n"


def _status(done: int = 100, total: int = 1000, **extra) -> RunStatus:
    return RunStatus(RunState.RUNNING, done, total, "2026-01-01T00:00:00", **extra)


class TestCheckpointReader:
    def test_should_summarize_every_checkpoint(self, tmp_path: Path):
        # Arrange
        path = tmp_path / "INDICATORS.csv"
        path.write_text(HEADER + BLOCK_100 + BLOCK_200)
        reader = CheckpointReader(path)

        # Act
        changed = reader.poll()

        # Assert
        assert changed
        first, second = reader.summaries
        assert (first.evaluation, first.size) == (100, 3)
        assert first.minimum == {"EP": 3.0, "NHV": 0.7}
        assert first.median == {"EP": 4.0, "NHV": 0.8}
        assert first.maximum == {"EP": 5.0, "NHV": 0.9}
        assert (second.evaluation, second.size) == (200, 2)
        assert reader.indicators == ["EP", "NHV"]

    def test_should_follow_the_file_as_it_grows(self, tmp_path: Path):
        # Arrange
        path = tmp_path / "INDICATORS.csv"
        path.write_text(HEADER + BLOCK_100)
        reader = CheckpointReader(path)
        reader.poll()

        # Act
        with path.open("a") as file:
            file.write(BLOCK_200)
        changed = reader.poll()

        # Assert
        assert changed
        assert [s.evaluation for s in reader.summaries] == [100, 200]
        assert not reader.poll()  # nothing new

    def test_should_wait_for_a_line_that_is_not_ended_yet(self, tmp_path: Path):
        # Arrange: Evolver is in the middle of writing the second line of the checkpoint 200
        path = tmp_path / "INDICATORS.csv"
        path.write_text(HEADER + BLOCK_100 + "200,0,2.0,0.5\n200,1,1.")
        reader = CheckpointReader(path)

        # Act
        reader.poll()

        # Assert: the unfinished line is not read, nor lost
        assert reader.summaries[-1].size == 1
        with path.open("a") as file:
            file.write("0,0.4\n")
        reader.poll()
        assert reader.summaries[-1].size == 2

    def test_should_start_a_checkpoint_where_the_solution_id_goes_back_to_zero(self, tmp_path):
        # Arrange: two checkpoints with the same evaluation (a final write that repeats it)
        path = tmp_path / "INDICATORS.csv"
        path.write_text(HEADER + BLOCK_100 + BLOCK_100)
        reader = CheckpointReader(path)

        # Act
        reader.poll()

        # Assert
        assert [s.size for s in reader.summaries] == [3, 3]

    def test_should_give_the_first_and_the_latest_checkpoint(self, tmp_path: Path):
        # Arrange
        path = tmp_path / "INDICATORS.csv"
        path.write_text(HEADER + BLOCK_100 + BLOCK_200 + BLOCK_300)
        reader = CheckpointReader(path)

        # Act
        reader.poll()

        # Assert
        assert reader.first is not None and set(reader.first["Evaluation"]) == {100}
        assert reader.latest is not None and set(reader.latest["Evaluation"]) == {300}
        assert reader.checkpoint(200) is not None
        assert reader.checkpoint(999) is None

    def test_should_keep_only_the_latest_checkpoints_in_memory(self, tmp_path: Path):
        # Arrange
        path = tmp_path / "INDICATORS.csv"
        rows = "".join(f"{100 * n},0,{n}.0,0.5\n" for n in range(1, 11))
        path.write_text(HEADER + rows)
        reader = CheckpointReader(path, kept_recent=3)

        # Act
        reader.poll()

        # Assert: every checkpoint is summarized, but only the first and the last are kept whole:
        # the 3 latest closed ones and the one being written
        assert len(reader.summaries) == 10
        assert len(reader.recent()) == 4
        assert reader.checkpoint(100) is not None  # the first is kept apart
        assert reader.checkpoint(500) is None
        assert reader.checkpoint(700) is not None
        assert reader.checkpoint(1000) is not None

    def test_should_do_nothing_before_the_file_exists(self, tmp_path: Path):
        # Arrange
        reader = CheckpointReader(tmp_path / "INDICATORS.csv")

        # Act / Assert
        assert not reader.poll()
        assert reader.summaries == [] and reader.latest is None and reader.first is None

    def test_should_start_again_if_the_file_is_replaced_by_a_shorter_one(self, tmp_path: Path):
        # Arrange
        path = tmp_path / "INDICATORS.csv"
        path.write_text(HEADER + BLOCK_100 + BLOCK_200)
        reader = CheckpointReader(path)
        reader.poll()

        # Act
        path.write_text(HEADER + BLOCK_300)
        reader.poll()

        # Assert
        assert [s.evaluation for s in reader.summaries] == [300]


VAR_CONF = """# Evaluation: 100
# Time (min): 0.5
EP=9 NHV=1 | --crossover SBX

# Evaluation: 200
# Time (min): 1.25
EP=5 NHV=1 | --crossover PCX
EP=4 NHV=2 | --crossover blxAlpha

"""


class TestVarConfFollower:
    def test_should_read_the_time_of_each_checkpoint_and_the_latest_front(self, tmp_path: Path):
        # Arrange
        path = tmp_path / "VAR_CONF.txt"
        path.write_text(VAR_CONF)
        follower = VarConfFollower(path)

        # Act
        changed = follower.poll()

        # Assert
        assert changed
        assert follower.minutes == {100: 0.5, 200: 1.25}
        assert follower.latest is not None
        assert [c.configuration for c in follower.latest.configurations] == [
            "--crossover PCX",
            "--crossover blxAlpha",
        ]

    def test_should_read_a_checkpoint_only_when_it_is_complete(self, tmp_path: Path):
        # Arrange: the checkpoint 300 is being written
        path = tmp_path / "VAR_CONF.txt"
        path.write_text(VAR_CONF + "# Evaluation: 300\n# Time (min): 2.0\nEP=3 NHV=1 | --cross")
        follower = VarConfFollower(path)

        # Act
        follower.poll()

        # Assert
        assert 300 not in follower.minutes
        with path.open("a") as file:
            file.write("over SBX\n\n")
        follower.poll()
        assert follower.minutes[300] == 2.0
        assert follower.latest is not None
        assert follower.latest.configurations[0].configuration == "--crossover SBX"

    def test_should_do_nothing_before_the_file_exists(self, tmp_path: Path):
        # Act / Assert
        follower = VarConfFollower(tmp_path / "VAR_CONF.txt")
        assert not follower.poll() and follower.latest is None


class TestSummarizeRun:
    START = datetime(2026, 1, 1, 10, 0, 0)

    def test_should_measure_the_pace_from_the_polls_and_estimate_the_time_left(self):
        # Arrange: 600 evaluations in 60 s of polls, 400 of 1000 to go
        samples = [(0.0, 400), (30.0, 700), (60.0, 1000)]
        status = _status(done=1000, total=2000)

        # Act
        summary = summarize_run(
            status, self.START, samples, self.START + timedelta(minutes=10), 60.0, True
        )

        # Assert
        assert summary.elapsed_seconds == 600.0
        assert summary.evaluations_per_minute == pytest.approx(600.0)
        assert summary.remaining_seconds == pytest.approx(100.0)
        assert not summary.stuck

    def test_should_fall_back_on_the_whole_run_before_it_has_two_polls(self):
        # Act
        summary = summarize_run(
            _status(done=300, total=1000),
            self.START,
            [(0.0, 300)],
            self.START + timedelta(minutes=10),
            0.0,
            True,
        )

        # Assert
        assert summary.evaluations_per_minute == pytest.approx(30.0)

    def test_should_take_the_time_left_from_the_limit_of_a_run_stopped_by_time(self):
        # Arrange
        status = _status(done=500, total=0, max_computing_time_minutes=10.0, elapsed_minutes=2.5)

        # Act
        summary = summarize_run(status, None, [], self.START, 0.0, True)

        # Assert
        assert summary.elapsed_seconds == 150.0
        assert summary.evaluations_total is None
        assert summary.remaining_seconds == pytest.approx(450.0)

    def test_should_not_know_what_it_cannot_measure(self):
        # Act
        summary = summarize_run(_status(done=0, total=1000), None, [], self.START, 0.0, True)

        # Assert
        assert summary.elapsed_seconds is None
        assert summary.evaluations_per_minute is None
        assert summary.remaining_seconds is None

    def test_should_say_a_run_is_stuck_when_it_stops_progressing_for_much_longer_than_usual(self):
        # Arrange: it progressed every 30 s, and has not for 20 minutes
        samples = [(0.0, 100), (30.0, 200), (60.0, 300), (90.0, 400), (1290.0, 400)]

        # Act
        summary = summarize_run(
            _status(done=400), self.START, samples, self.START + timedelta(minutes=22), 1290.0, True
        )

        # Assert
        assert summary.stuck
        assert summary.seconds_without_progress == pytest.approx(1200.0)

    def test_should_not_call_a_run_stuck_after_a_normal_pause_or_with_its_process_gone(self):
        # Arrange
        samples = [(0.0, 100), (30.0, 200), (60.0, 300), (100.0, 300)]
        now = self.START + timedelta(minutes=2)

        # Act
        slow = summarize_run(_status(done=300), self.START, samples, now, 100.0, True)
        gone = summarize_run(_status(done=300), self.START, samples, now, 5000.0, False)

        # Assert
        assert not slow.stuck
        assert not gone.stuck

    @pytest.mark.parametrize(
        ("seconds", "text"),
        [(None, "—"), (9, "9 s"), (724, "12 min 4 s"), (7500, "2 h 5 min")],
    )
    def test_should_write_a_duration(self, seconds, text):
        # Act / Assert
        assert format_duration(seconds) == text


def _summaries(count: int) -> list[CheckpointSummary]:
    return [
        CheckpointSummary(
            evaluation=100 * (i + 1),
            size=3,
            minimum={"EP": 1.0 / (i + 1), "NHV": 0.5},
            median={"EP": 2.0 / (i + 1), "NHV": 0.6},
            maximum={"EP": 3.0 / (i + 1), "NHV": 0.7},
        )
        for i in range(count)
    ]


def _rows(evaluation: int, points: list[tuple[float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Evaluation": evaluation,
            "SolutionId": range(len(points)),
            "EP": [p[0] for p in points],
            "NHV": [p[1] for p in points],
        }
    )


class TestPopulationFigure:
    def test_should_draw_the_first_front_the_population_and_the_front(self):
        # Arrange
        population = _rows(300, [(1, 1), (2, 2), (3, 3)])
        front = _rows(300, [(1, 1)])
        first = _rows(100, [(9, 9)])

        # Act
        figure = build_population_figure(population, front, first, ("EP", "NHV"))

        # Assert: faded first front underneath, then the population, then the front on top
        assert [trace.name for trace in figure.data] == [
            "First checkpoint",
            "Population",
            "Non-dominated front",
        ]
        assert figure.layout.title.text == "Population at evaluation 300"
        assert figure.layout.xaxis.title.text == "EP"

    def test_should_draw_only_the_front_when_the_run_did_not_write_its_population(self):
        # Act
        figure = build_population_figure(None, _rows(300, [(1, 1)]), None, ("EP", "NHV"))

        # Assert
        assert [trace.name for trace in figure.data] == ["Non-dominated front"]
        assert "evaluation 300" in figure.layout.title.text

    def test_should_use_logarithmic_axes_on_request(self):
        # Act
        figure = build_population_figure(
            _rows(300, [(1, 1)]), None, None, ("EP", "NHV"), log_scale=True
        )

        # Assert
        assert figure.layout.xaxis.type == "log" and figure.layout.yaxis.type == "log"


class TestConvergenceFigure:
    def test_should_draw_a_panel_per_indicator_with_the_best_the_median_and_the_worst(self):
        # Act
        figure = build_convergence_figure(_summaries(5))

        # Assert: per indicator, the upper edge of the band, the best (filled) and the median
        assert len(figure.data) == 6
        assert list(figure.data[1].x) == [100, 200, 300, 400, 500]
        assert figure.layout.yaxis.title.text == "EP"
        assert figure.layout.yaxis2.title.text == "NHV"
        assert figure.layout.xaxis2.title.text == "Meta-evaluations"

    def test_should_put_the_computing_time_on_the_x_axis_when_it_is_known(self):
        # Arrange: the time of the checkpoint 300 is not known (yet)
        minutes = {100: 0.5, 200: 1.0, 400: 2.0, 500: 2.5}

        # Act
        figure = build_convergence_figure(_summaries(5), minutes=minutes)

        # Assert
        assert list(figure.data[1].x) == [0.5, 1.0, 2.0, 2.5]
        assert figure.layout.xaxis2.title.text == "Computing time (min)"

    def test_should_thin_a_very_long_history_keeping_the_last_checkpoint(self):
        # Act
        figure = build_convergence_figure(_summaries(10_000))

        # Assert
        points = list(figure.data[1].x)
        assert len(points) <= 2001
        assert points[-1] == 1_000_000

    def test_should_draw_an_empty_figure_without_checkpoints(self):
        # Act / Assert
        assert len(build_convergence_figure([]).data) == 0
