"""Tests for the throttled live indicator-front preview."""

from pathlib import Path

import pandas as pd
import pytest

from evolver_studio.live_front import (
    LiveFrontRenderer,
    build_front_figure,
    should_render_checkpoint,
)

CSV_HEADER = "Evaluation,SolutionId,Epsilon,NormalizedHypervolume\n"


def _checkpoint_row(evaluation: int) -> str:
    return f"{evaluation},0,0.5,0.8\n"


class TestBuildFrontFigure:
    def test_should_plot_one_point_per_row_colored_by_evaluation(self):
        """Every checkpoint's rows must appear, with color driven by Evaluation."""
        # Arrange
        history = pd.DataFrame(
            {
                "Evaluation": [50, 200],
                "SolutionId": [0, 0],
                "Epsilon": [0.5, 0.2],
                "NHV": [0.8, 0.95],
            }
        )

        # Act
        figure = build_front_figure(history)

        # Assert
        assert list(figure.data[0].x) == [0.5, 0.2]
        assert list(figure.data[0].marker.color) == [50, 200]


class TestShouldRenderCheckpoint:
    @pytest.mark.parametrize(
        ("latest_evaluation", "last_rendered_evaluation", "update_every_evaluations", "expected"),
        [
            (100, None, 100, True),
            (300, 100, 100, True),
            (150, 100, 100, False),
            (200, 100, 100, True),
        ],
        ids=["first-checkpoint", "advanced-past-n", "advanced-less-than-n", "advanced-exactly-n"],
    )
    def test_should_decide_whether_checkpoint_is_due(
        self, latest_evaluation, last_rendered_evaluation, update_every_evaluations, expected
    ):
        """Throttling must fire on the first checkpoint and every N evaluations after."""
        # Act
        is_due = should_render_checkpoint(
            latest_evaluation, last_rendered_evaluation, update_every_evaluations
        )

        # Assert
        assert is_due is expected


class TestLiveFrontRenderer:
    def test_should_report_no_change_when_file_does_not_exist_yet(self, tmp_path: Path):
        """Polling before Evolver has written anything must not raise."""
        # Arrange
        renderer = LiveFrontRenderer(update_every_evaluations=100)
        indicators_csv = tmp_path / "missing.csv"

        # Act
        update = renderer.poll(indicators_csv)

        # Assert
        assert update.changed is False
        assert update.updated is False
        assert renderer.last_history is None

    def test_should_fold_in_first_checkpoint(self, tmp_path: Path):
        """The first checkpoint seen is always due, regardless of N."""
        # Arrange
        renderer = LiveFrontRenderer(update_every_evaluations=100)
        indicators_csv = tmp_path / "INDICATORS.csv"
        indicators_csv.write_text(CSV_HEADER + _checkpoint_row(50))

        # Act
        update = renderer.poll(indicators_csv)

        # Assert
        assert update.changed is True
        assert update.updated is True
        assert renderer.last_rendered_evaluation == 50

    def test_should_report_no_change_when_file_unchanged(self, tmp_path: Path):
        """Polling again with no new bytes written must skip re-parsing and redrawing."""
        # Arrange
        renderer = LiveFrontRenderer(update_every_evaluations=100)
        indicators_csv = tmp_path / "INDICATORS.csv"
        indicators_csv.write_text(CSV_HEADER + _checkpoint_row(50))
        renderer.poll(indicators_csv)

        # Act
        update = renderer.poll(indicators_csv)

        # Assert
        assert update.changed is False
        assert update.updated is False

    def test_should_report_change_but_skip_update_before_threshold_is_reached(self, tmp_path: Path):
        """A new checkpoint that hasn't advanced by N evaluations yet is skipped."""
        # Arrange
        renderer = LiveFrontRenderer(update_every_evaluations=100)
        indicators_csv = tmp_path / "INDICATORS.csv"
        indicators_csv.write_text(CSV_HEADER + _checkpoint_row(50))
        renderer.poll(indicators_csv)
        with indicators_csv.open("a") as handle:
            handle.write(_checkpoint_row(90))

        # Act
        update = renderer.poll(indicators_csv)

        # Assert
        assert update.changed is True
        assert update.updated is False
        assert renderer.last_rendered_evaluation == 50

    def test_should_update_again_once_threshold_is_reached(self, tmp_path: Path):
        """A checkpoint at least N evaluations past the last render is due."""
        # Arrange
        renderer = LiveFrontRenderer(update_every_evaluations=100)
        indicators_csv = tmp_path / "INDICATORS.csv"
        indicators_csv.write_text(CSV_HEADER + _checkpoint_row(50))
        renderer.poll(indicators_csv)
        with indicators_csv.open("a") as handle:
            handle.write(_checkpoint_row(160))

        # Act
        update = renderer.poll(indicators_csv)

        # Assert
        assert update.updated is True
        assert renderer.last_rendered_evaluation == 160

    def test_should_keep_last_history_available_across_unchanged_polls(self, tmp_path: Path):
        """A caller redrawing every tick (e.g. inside a fragment) must not lose the history."""
        # Arrange
        renderer = LiveFrontRenderer(update_every_evaluations=100)
        indicators_csv = tmp_path / "INDICATORS.csv"
        indicators_csv.write_text(CSV_HEADER + _checkpoint_row(50))
        renderer.poll(indicators_csv)

        # Act
        renderer.poll(indicators_csv)

        # Assert
        assert renderer.last_history is not None
        assert renderer.last_rendered_evaluation == 50

    def test_should_accumulate_every_distinct_checkpoint_seen_so_far(self, tmp_path: Path):
        """last_history must include earlier checkpoints too, not just the latest."""
        # Arrange
        renderer = LiveFrontRenderer(update_every_evaluations=100)
        indicators_csv = tmp_path / "INDICATORS.csv"
        indicators_csv.write_text(CSV_HEADER + "50,0,0.5,0.8\n")
        renderer.poll(indicators_csv)
        with indicators_csv.open("a") as handle:
            handle.write("200,0,0.2,0.95\n")

        # Act
        renderer.poll(indicators_csv)

        # Assert
        assert sorted(renderer.last_history["Evaluation"].unique()) == [50, 200]
