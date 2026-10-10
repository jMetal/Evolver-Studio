"""Tests for the empirical attainment functions of a validation study."""

import pandas as pd
import pytest

from evolver_studio.attainment import (
    BEST,
    MEDIAN,
    WORST,
    attainment_surfaces,
    eaf_differences,
)


def _front(*points: tuple[float, float]) -> pd.DataFrame:
    return pd.DataFrame(points, columns=["f1", "f2"])


# Two runs: the first reaches (1, 2) and (2, 1); the second, worse, only (2, 3).
RUNS = {1: _front((1, 2), (2, 1)), 2: _front((2, 3))}


class TestAttainmentSurfaces:
    def test_should_give_the_points_reached_by_one_run_and_by_all(self):
        # Act
        surfaces = attainment_surfaces(RUNS)

        # Assert: at least one run reaches the first run's front; both reach only (2, 3)
        assert surfaces[BEST].values.tolist() == [[1, 2], [2, 1]]
        assert surfaces[WORST].values.tolist() == [[2, 3]]
        assert MEDIAN in surfaces

    def test_should_give_nothing_without_fronts(self):
        # Act / Assert
        assert attainment_surfaces({}) == {}


class TestEafDifferences:
    def test_should_find_where_the_left_runs_reach_more_often(self):
        # Arrange: the right algorithm's runs both reach only (3, 3)
        right = {1: _front((3, 3)), 2: _front((3, 3))}

        # Act
        differences = eaf_differences(RUNS, right)

        # Assert: the left is ahead somewhere, never behind, by at most all of its runs
        assert not differences.empty
        assert (differences["difference"] > 0).all()
        assert differences["difference"].max() == pytest.approx(1.0)

    def test_should_find_no_difference_between_the_same_runs(self):
        # Act / Assert
        assert eaf_differences(RUNS, RUNS).empty
