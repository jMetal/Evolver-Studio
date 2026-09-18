"""Tests for the provisional algorithm catalogue."""

from pathlib import Path

import pytest

from evolver_studio.catalogue import (
    BASE_ALGORITHMS,
    KNOWN_NON_ALGORITHM_PARAMETER_SPACE_FILES,
    META_ALGORITHMS,
)
from evolver_studio.evolver_client import describe, jar_path
from evolver_studio.result import Ok

EVOLVER_HOME = Path("/Users/ajnebro/Softw/Evolver")
EVOLVER_PARAMETER_SPACES_DIR = EVOLVER_HOME / "src/main/resources/parameterSpaces"


class TestBaseAlgorithms:
    def test_should_have_unique_names(self):
        """Two entries with the same name would make lookups ambiguous."""
        # Act
        names = [algorithm.name for algorithm in BASE_ALGORITHMS]

        # Assert
        assert len(names) == len(set(names))

    def test_should_have_at_least_one_encoding_each(self):
        """An algorithm with no encodings would have nothing to explore or run."""
        # Act / Assert
        assert all(algorithm.encodings for algorithm in BASE_ALGORITHMS)

    def test_should_mark_only_nsgaii_and_moead_as_runnable_today(self):
        """BaseAlgorithmRegistry in cli.runner only resolves these two names."""
        # Act
        runnable = {a.name for a in BASE_ALGORITHMS if a.runnable_today}

        # Assert
        assert runnable == {"NSGA-II", "MOEA/D"}

    def test_should_set_registry_name_and_runnable_encoding_iff_runnable_today(self):
        """A runnable algorithm needs both to build a request; a non-runnable one needs neither."""
        # Act / Assert
        for algorithm in BASE_ALGORITHMS:
            has_registry_fields = (
                algorithm.registry_name is not None and algorithm.runnable_encoding is not None
            )
            assert has_registry_fields is algorithm.runnable_today

    def test_should_use_the_registry_key_for_moead_not_the_display_name(self):
        """BaseAlgorithmRegistry.resolve() expects "MOEAD", not the display name "MOEA/D"."""
        # Arrange
        moead = next(a for a in BASE_ALGORITHMS if a.name == "MOEA/D")

        # Act / Assert
        assert moead.registry_name == "MOEAD"

    def test_should_make_runnable_encoding_a_real_encoding_of_the_algorithm(self):
        """A typo'd runnable_encoding would point at a parameter space file that doesn't exist."""
        # Act / Assert
        for algorithm in BASE_ALGORITHMS:
            if algorithm.runnable_encoding is not None:
                assert algorithm.runnable_encoding in algorithm.encodings


class TestMetaAlgorithms:
    def test_should_have_unique_names(self):
        """Two entries with the same name would make lookups ambiguous."""
        # Act
        names = [algorithm.name for algorithm in META_ALGORITHMS]

        # Assert
        assert len(names) == len(set(names))

    def test_should_support_at_least_one_encoding_each(self):
        """A meta-algorithm supporting neither encoding could never be selected."""
        # Act / Assert
        assert all(a.supports_flat or a.supports_tree for a in META_ALGORITHMS)

    def test_should_have_tree_parameters_only_when_tree_is_supported(self):
        """A non-empty tree_parameters on a flat-only algorithm would be misleading."""
        # Act / Assert
        assert all(a.supports_tree or not a.tree_parameters for a in META_ALGORITHMS)

    def test_should_mark_smpso_as_not_supporting_tree(self):
        """SMPSO's velocity-based representation is structurally incompatible with tree encoding."""
        # Arrange
        smpso = next(a for a in META_ALGORITHMS if a.name == "SMPSO")

        # Act / Assert
        assert smpso.supports_tree is False

    def test_should_mark_the_four_registered_meta_algorithms_as_wired(self):
        """MetaAlgorithmRegistry registers exactly these four for the flat encoding."""
        # Act
        wired = {a.name for a in META_ALGORITHMS if a.wired_into_cli_runner}

        # Assert
        assert wired == {"NSGA-II", "SPEA2", "SMPSO", "AsyncNSGA-II"}

    def test_should_mark_only_parallel_nsgaii_as_supporting_tree_among_wired_algorithms(self):
        """MetaAlgorithmRegistry.validateTreeAlgorithm only accepts NSGA-II."""
        # Act
        tree_wired = {
            a.name for a in META_ALGORITHMS if a.wired_into_cli_runner and a.supports_tree
        }

        # Assert
        assert tree_wired == {"NSGA-II"}


class TestCatalogueMatchesEvolverCheckout:
    def test_should_reference_parameter_space_files_that_actually_exist(self):
        """A stale filename in the catalogue would break the explorer at browse time."""
        if not EVOLVER_PARAMETER_SPACES_DIR.is_dir():
            pytest.skip(f"Evolver checkout not found at {EVOLVER_PARAMETER_SPACES_DIR}")

        # Arrange
        missing = [
            f"{algorithm.name}/{encoding}: {filename}"
            for algorithm in BASE_ALGORITHMS
            for encoding, filename in algorithm.encodings.items()
            if not (EVOLVER_PARAMETER_SPACES_DIR / filename).is_file()
        ]

        # Assert
        assert missing == []

    def test_should_reference_meta_operator_parameter_space_files_that_actually_exist(self):
        """A stale operator_parameter_space_file would break the explorer at browse time."""
        if not EVOLVER_PARAMETER_SPACES_DIR.is_dir():
            pytest.skip(f"Evolver checkout not found at {EVOLVER_PARAMETER_SPACES_DIR}")

        # Arrange
        missing = [
            f"{algorithm.name}: {algorithm.operator_parameter_space_file}"
            for algorithm in META_ALGORITHMS
            if algorithm.operator_parameter_space_file is not None
            and not (
                EVOLVER_PARAMETER_SPACES_DIR / algorithm.operator_parameter_space_file
            ).is_file()
        ]

        # Assert
        assert missing == []

    def test_should_flag_any_untriaged_parameter_space_file(self):
        """A new YAML file (new algorithm, new encoding) must not go unnoticed.

        Evolver-Studio has no way to learn about a new algorithm on its own —
        every file under parameterSpaces/ must be accounted for, either in
        BASE_ALGORITHMS (a real base-level algorithm to add to the catalogue)
        or in KNOWN_NON_ALGORITHM_PARAMETER_SPACE_FILES (deliberately not
        one, with a reason in that constant's comments). This is the
        Evolver-Studio half of the drift-tracking mechanism described in
        CLAUDE.md; org.uma.evolver.cli.runner has a companion Java test.
        """
        if not EVOLVER_PARAMETER_SPACES_DIR.is_dir():
            pytest.skip(f"Evolver checkout not found at {EVOLVER_PARAMETER_SPACES_DIR}")

        # Arrange
        known = {
            filename for algorithm in BASE_ALGORITHMS for filename in algorithm.encodings.values()
        } | KNOWN_NON_ALGORITHM_PARAMETER_SPACE_FILES
        actual = {path.name for path in EVOLVER_PARAMETER_SPACES_DIR.iterdir() if path.is_file()}

        # Act
        untriaged = sorted(actual - known)

        # Assert
        assert untriaged == [], (
            f"New file(s) under parameterSpaces/ not yet triaged: {untriaged}. Add each one to "
            "BASE_ALGORITHMS (if it's a real algorithm's parameter space) or to "
            "KNOWN_NON_ALGORITHM_PARAMETER_SPACE_FILES (with a reason) in "
            "evolver_studio/catalogue.py."
        )


class TestCatalogueMatchesDescribeManifest:
    """Cross-checks the hand-maintained catalogue against Evolver's own introspection manifest.

    See Evolver's docs/proposals/cli-describe-manifest.md: DescribeMain lists exactly what
    BaseAlgorithmRegistry/MetaAlgorithmRegistry register, generated from their own data, so it
    cannot itself drift — this test instead catches *this* module going stale relative to it.
    Skipped if the checkout/jar isn't available, same as the other checkout-dependent tests here.
    """

    def _manifest(self) -> dict | None:
        jar = jar_path(EVOLVER_HOME)
        if not jar.is_file():
            return None
        result = describe(EVOLVER_HOME, jar)
        return result.value if isinstance(result, Ok) else None

    def test_should_have_a_wired_meta_algorithm_entry_for_every_manifest_entry(self):
        """Every algorithm DescribeMain lists as registered must be wired=True here too."""
        manifest = self._manifest()
        if manifest is None:
            pytest.skip(f"Evolver jar not found under {EVOLVER_HOME}")

        # Arrange
        manifest_names = {a["name"] for a in manifest["metaAlgorithms"]}
        wired_names = {a.name for a in META_ALGORITHMS if a.wired_into_cli_runner}

        # Assert
        assert manifest_names == wired_names

    def test_should_have_a_runnable_base_algorithm_entry_for_every_manifest_entry(self):
        """Every algorithm DescribeMain lists as registered must be runnable_today=True here too."""
        manifest = self._manifest()
        if manifest is None:
            pytest.skip(f"Evolver jar not found under {EVOLVER_HOME}")

        # Arrange
        manifest_names = {a["name"] for a in manifest["baseAlgorithms"]}
        runnable_names = {a.registry_name for a in BASE_ALGORITHMS if a.runnable_today}

        # Assert
        assert manifest_names == runnable_names
