"""Tests for estimating the time a run has left."""

import pytest

from evolver_studio.progress import estimate_remaining_seconds, format_duration


class TestEstimateRemainingSeconds:
    def test_should_extrapolate_the_pace_so_far(self):
        # Act: a quarter done in 10 s
        remaining = estimate_remaining_seconds(10.0, 2500, 10000)

        # Assert
        assert remaining == pytest.approx(30.0)

    def test_should_have_no_estimate_before_anything_is_done(self):
        # Act / Assert
        assert estimate_remaining_seconds(5.0, 0, 10000) is None
        assert estimate_remaining_seconds(0.0, 100, 10000) is None

    def test_should_have_nothing_left_when_the_evaluations_are_done(self):
        # Act / Assert
        assert estimate_remaining_seconds(10.0, 10000, 10000) == 0.0


class TestFormatDuration:
    @pytest.mark.parametrize(
        ("seconds", "text"),
        [
            (0, "0 s"),
            (7.4, "7 s"),
            (59.6, "1 min"),
            (60, "1 min"),
            (125, "2 min 5 s"),
            (3600, "1 h"),
            (3780, "1 h 3 min"),
        ],
    )
    def test_should_write_the_largest_units(self, seconds: float, text: str):
        # Act / Assert
        assert format_duration(seconds) == text
