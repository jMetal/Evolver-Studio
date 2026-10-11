"""Tests for the base-level choices of a training: meta-objectives, runs and population."""

from pathlib import Path

from evolver_studio.training_choices import (
    indicator_errors,
    population_errors,
    training_indicators,
)


class TestIndicators:
    def test_should_offer_only_indicators_evolver_computes_and_minimizes(self):
        # Act
        names = [indicator.registry_name for indicator in training_indicators()]

        # Assert: the hypervolume Evolver-Studio computes for a validation is not one
        assert "Hypervolume" not in names
        assert {"Epsilon", "NormalizedHypervolume", "HypervolumeMinus"} <= set(names)

    def test_should_ask_for_two_meta_objectives(self):
        # Act / Assert
        assert indicator_errors(("Epsilon", "NormalizedHypervolume"), [2]) == []
        assert indicator_errors(("Epsilon",), [2])

    def test_should_take_spread_only_for_problems_with_two_objectives(self):
        # Act / Assert
        assert indicator_errors(("Epsilon", "Spread"), [2, 2]) == []
        assert "GeneralizedSpread" in indicator_errors(("Epsilon", "Spread"), [2, 3])[0]


class TestPopulation:
    def test_should_need_a_weight_vector_file_for_every_number_of_objectives(self, tmp_path: Path):
        # Arrange
        for name in ("W2D_100.dat", "W3D_91.dat"):
            (tmp_path / name).write_text("")

        # Act / Assert
        assert population_errors(100, True, [2, 2], tmp_path) == []
        errors = population_errors(100, True, [2, 3], tmp_path)
        assert len(errors) == 1 and "91" in errors[0]

    def test_should_take_any_size_for_an_algorithm_without_weight_vectors(self, tmp_path: Path):
        # Act / Assert
        assert population_errors(37, False, [3], tmp_path) == []
