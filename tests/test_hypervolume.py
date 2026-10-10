"""Tests for the hypervolume computed as Evolver computes its hypervolume indicators."""

import numpy as np
import pytest

from evolver_studio.hypervolume import hypervolume

# The reference front spans [0, 2] x [0, 4]: fronts are normalized with it.
REFERENCE_FRONT = np.array([[0.0, 4.0], [1.0, 2.0], [2.0, 0.0]])


class TestHypervolume:
    def test_should_measure_up_to_the_point_one_after_normalizing_with_the_reference_front(self):
        # Act: (1, 2) normalizes to (0.5, 0.5)
        value = hypervolume(np.array([[1.0, 2.0]]), REFERENCE_FRONT)

        # Assert
        assert value == pytest.approx(0.25)

    def test_should_not_count_dominated_solutions(self):
        # Act
        alone = hypervolume(np.array([[1.0, 2.0]]), REFERENCE_FRONT)
        with_dominated = hypervolume(np.array([[1.0, 2.0], [1.5, 3.0]]), REFERENCE_FRONT)

        # Assert
        assert with_dominated == pytest.approx(alone)

    def test_should_clip_what_falls_outside_the_reference_front_bounds(self):
        # Act: beyond the upper bound in one objective adds nothing; below 0 counts as 0
        outside = hypervolume(np.array([[3.0, 0.0]]), REFERENCE_FRONT)
        below = hypervolume(np.array([[-1.0, 2.0]]), REFERENCE_FRONT)

        # Assert
        assert outside == pytest.approx(0.0)
        assert below == pytest.approx(0.5)

    def test_should_measure_three_objectives(self):
        # Arrange
        reference_front = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

        # Act
        value = hypervolume(np.array([[0.5, 0.5, 0.5]]), reference_front)

        # Assert
        assert value == pytest.approx(0.125)

    def test_should_give_zero_for_an_empty_front(self):
        # Act / Assert
        assert hypervolume(np.empty((0, 2)), REFERENCE_FRONT) == 0.0

    def test_should_reject_fronts_with_different_objectives(self):
        # Act / Assert
        with pytest.raises(ValueError, match="objectives"):
            hypervolume(np.array([[0.5, 0.5, 0.5]]), REFERENCE_FRONT)
