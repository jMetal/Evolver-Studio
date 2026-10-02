"""Tests for finding a problem's reference front files."""

from pathlib import Path

import pytest

from evolver_studio.problems import (
    default_reference_front,
    reference_front_candidates,
    reference_front_dimension,
)


@pytest.fixture
def resources(tmp_path: Path) -> Path:
    """A resources/ directory with a few of Evolver's reference fronts."""
    root = tmp_path / "resources"
    (root / "referenceFronts").mkdir(parents=True)
    (root / "referenceFrontsTSP").mkdir()
    for name in (
        "ZDT1.csv",
        "ZDT1.ReferencePoints.csv",
        "ZDT10.csv",
        "DTLZ2.8D.csv",
        "DTLZ2.3D.csv",
        "DTLZ2.2D.csv",
    ):
        (root / "referenceFronts" / name).write_text("0,1\n")
    (root / "referenceFrontsTSP" / "KroAB100TSP.csv").write_text("0,1\n")
    return root


class TestReferenceFrontCandidates:
    def test_should_find_the_front_of_a_problem_with_a_single_one(self, resources: Path):
        # Act
        candidates = reference_front_candidates("ZDT1", resources)

        # Assert: not ZDT10, nor ZDT1's reference points
        assert candidates == ["resources/referenceFronts/ZDT1.csv"]

    def test_should_order_the_fronts_by_number_of_objectives(self, resources: Path):
        # Act
        candidates = reference_front_candidates("DTLZ2", resources)

        # Assert
        assert candidates == [
            "resources/referenceFronts/DTLZ2.2D.csv",
            "resources/referenceFronts/DTLZ2.3D.csv",
            "resources/referenceFronts/DTLZ2.8D.csv",
        ]

    def test_should_find_a_tsp_front_from_a_fully_qualified_class_name(self, resources: Path):
        # Act
        candidates = reference_front_candidates(
            "org.uma.jmetal.problem.multiobjective.multiobjectivetsp.instance.KroAB100TSP",
            resources,
        )

        # Assert
        assert candidates == ["resources/referenceFrontsTSP/KroAB100TSP.csv"]

    def test_should_find_nothing_for_an_unknown_problem(self, resources: Path):
        # Act / Assert
        assert reference_front_candidates("NoSuchProblem", resources) == []

    def test_should_find_the_real_fronts_copied_from_evolver(self):
        # Act
        candidates = reference_front_candidates("DTLZ2")

        # Assert
        assert "resources/referenceFronts/DTLZ2.3D.csv" in candidates


class TestDefaultReferenceFront:
    def test_should_pick_the_only_candidate(self):
        # Act / Assert
        assert default_reference_front(["resources/referenceFronts/ZDT1.csv"]) == (
            "resources/referenceFronts/ZDT1.csv"
        )

    @pytest.mark.parametrize("candidates", [[], ["a.2D.csv", "a.3D.csv"]])
    def test_should_not_pick_when_there_is_none_or_a_doubt(self, candidates: list[str]):
        # Act / Assert
        assert default_reference_front(candidates) is None


class TestReferenceFrontDimension:
    def test_should_count_the_columns_of_the_first_line(self, tmp_path: Path):
        # Arrange
        front = tmp_path / "front.csv"
        front.write_text("0.1,0.2,0.3\n0.4,0.5,0.6\n")

        # Act / Assert
        assert reference_front_dimension(front) == 3

    def test_should_have_no_dimension_for_a_missing_or_empty_file(self, tmp_path: Path):
        # Arrange
        empty = tmp_path / "empty.csv"
        empty.write_text("")

        # Act / Assert
        assert reference_front_dimension(empty) is None
        assert reference_front_dimension(tmp_path / "missing.csv") is None

    def test_should_read_the_real_front_of_dtlz2_with_three_objectives(self):
        # Act
        dimension = reference_front_dimension(
            Path(__file__).parent.parent / "resources/referenceFronts/DTLZ2.3D.csv"
        )

        # Assert
        assert dimension == 3
