"""Tests for the provisional algorithm catalogue."""

import zipfile

import pytest

from evolver_studio.catalogue import (
    BASE_ALGORITHMS,
    KNOWN_NON_ALGORITHM_PARAMETER_SPACE_FILES,
    META_ALGORITHMS,
    QUALITY_INDICATORS,
    is_older_than_catalogue,
)
from evolver_studio.evolver_client import WORKING_DIRECTORY, describe, jar_path
from evolver_studio.resource_files import PARAMETER_SPACES_DIRECTORY
from evolver_studio.result import Ok


def _jar_parameter_space_files() -> set[str]:
    """The parameter space file names packaged in the app's Evolver jar, skipping if absent."""
    jar = jar_path()
    if not jar.is_file():
        pytest.skip(f"Evolver jar not found at {jar}")
    with zipfile.ZipFile(jar) as archive:
        prefix = f"{PARAMETER_SPACES_DIRECTORY}/"
        return {
            name.removeprefix(prefix)
            for name in archive.namelist()
            if name.startswith(prefix) and name != prefix
        }


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

    def test_should_mark_only_nsgaii_moead_and_rvea_as_runnable_today(self):
        """BaseAlgorithmRegistry in cli.training only resolves these three names."""
        # Act
        runnable = {a.name for a in BASE_ALGORITHMS if a.runnable_today}

        # Assert
        assert runnable == {"NSGA-II", "MOEA/D", "RVEA"}

    def test_should_ask_for_weight_vectors_only_for_moead_and_rvea(self):
        """Both read their weight vectors from files, so both need the directory."""
        # Act
        needing = {
            a.name
            for a in BASE_ALGORITHMS
            if "weightVectorFilesDirectory" in a.required_extra_config_keys
        }

        # Assert
        assert needing == {"MOEA/D", "RVEA"}

    def test_should_set_registry_name_and_runnable_encodings_iff_runnable_today(self):
        """A runnable algorithm needs both to build a request; a non-runnable one needs neither."""
        # Act / Assert
        for algorithm in BASE_ALGORITHMS:
            has_registry_fields = algorithm.registry_name is not None and bool(
                algorithm.runnable_encodings
            )
            assert has_registry_fields is algorithm.runnable_today

    def test_should_use_the_registry_key_for_moead_not_the_display_name(self):
        """BaseAlgorithmRegistry.resolve() expects "MOEAD", not the display name "MOEA/D"."""
        # Arrange
        moead = next(a for a in BASE_ALGORITHMS if a.name == "MOEA/D")

        # Act / Assert
        assert moead.registry_name == "MOEAD"

    def test_should_make_runnable_encodings_real_encodings_of_the_algorithm(self):
        """A typo'd runnable_encodings entry would point at a nonexistent parameter space file."""
        # Act / Assert
        for algorithm in BASE_ALGORITHMS:
            for encoding in algorithm.runnable_encodings:
                assert encoding in algorithm.encodings

    def test_should_mark_nsgaii_as_runnable_for_both_double_and_permutation(self):
        """BaseAlgorithmRegistry now registers ("NSGA-II", "Permutation") too."""
        # Arrange
        nsgaii = next(a for a in BASE_ALGORITHMS if a.name == "NSGA-II")

        # Act / Assert
        assert set(nsgaii.runnable_encodings) == {"Double", "Permutation"}


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

    def test_should_mark_the_six_registered_meta_algorithms_as_wired(self):
        """MetaAlgorithmRegistry registers exactly these six for the flat encoding."""
        # Act
        wired = {a.name for a in META_ALGORITHMS if a.wired_into_cli_runner}

        # Assert
        assert wired == {"NSGA-II", "AGE-MOEA", "SPEA2", "SMPSO", "AsyncNSGA-II", "RandomSearch"}

    def test_should_mark_four_wired_meta_algorithms_as_supporting_tree(self):
        """MetaAlgorithmRegistry.validateTreeAlgorithm accepts exactly these four."""
        # Act
        tree_wired = {
            a.name for a in META_ALGORITHMS if a.wired_into_cli_runner and a.supports_tree
        }

        # Assert
        assert tree_wired == {"NSGA-II", "AGE-MOEA", "AsyncNSGA-II", "RandomSearch"}

    def test_should_have_a_tree_operator_catalogue_only_when_tree_is_supported(self):
        """A tree operator catalogue on a flat-only algorithm would never be used."""
        # Act / Assert
        assert all(
            a.supports_tree or a.tree_operator_parameter_space_file is None for a in META_ALGORITHMS
        )


class TestQualityIndicators:
    def test_should_have_unique_registry_and_short_names(self):
        """Two entries with the same name would make the indicators page ambiguous."""
        # Act
        registry_names = [indicator.registry_name for indicator in QUALITY_INDICATORS]
        short_names = [indicator.short_name for indicator in QUALITY_INDICATORS]

        # Assert
        assert len(registry_names) == len(set(registry_names))
        assert len(short_names) == len(set(short_names))


class TestIsOlderThanCatalogue:
    """The catalogue mirrors Evolver 2.2 (CATALOGUE_EVOLVER_VERSION)."""

    @pytest.mark.parametrize("version", ["2.1", "2.0", "1.0.1"])
    def test_should_flag_an_earlier_release(self, version: str):
        # Act / Assert
        assert is_older_than_catalogue(version)

    @pytest.mark.parametrize("version", ["2.2", "2.2-SNAPSHOT", "2.2.1", "2.3-SNAPSHOT", "3.0"])
    def test_should_accept_the_same_or_a_later_release_or_its_snapshot(self, version: str):
        """A snapshot built from develop on its way to 2.2 already has its catalogue."""
        # Act / Assert
        assert not is_older_than_catalogue(version)


class TestCatalogueMatchesEvolverJar:
    """The app reads parameter spaces from Evolver's jar (see resource_files.py)."""

    def test_should_reference_parameter_space_files_that_actually_exist(self):
        """A stale filename in the catalogue would break the explorer at browse time."""
        # Arrange
        available = _jar_parameter_space_files()

        # Act
        missing = [
            f"{algorithm.name}/{encoding}: {filename}"
            for algorithm in BASE_ALGORITHMS
            for encoding, filename in algorithm.encodings.items()
            if filename not in available
        ]

        # Assert
        assert missing == []

    def test_should_reference_meta_operator_parameter_space_files_that_actually_exist(self):
        """A stale operator_parameter_space_file would break the explorer at browse time."""
        # Arrange
        available = _jar_parameter_space_files()

        # Act
        missing = [
            f"{algorithm.name}: {filename}"
            for algorithm in META_ALGORITHMS
            for filename in (
                algorithm.operator_parameter_space_file,
                algorithm.tree_operator_parameter_space_file,
            )
            if filename is not None and filename not in available
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
        CLAUDE.md; org.uma.evolver.cli.training has companion Java tests. Run it against a
        jar built from Evolver's develop branch (EVOLVER_JAR) to catch changes before a release.
        """
        # Arrange
        actual = _jar_parameter_space_files()
        known = {
            filename for algorithm in BASE_ALGORITHMS for filename in algorithm.encodings.values()
        } | KNOWN_NON_ALGORITHM_PARAMETER_SPACE_FILES

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
    Skipped if the app's Evolver jar isn't available.
    """

    def _manifest(self) -> dict | None:
        jar = jar_path()
        if not jar.is_file():
            return None
        result = describe(WORKING_DIRECTORY, jar)
        return result.value if isinstance(result, Ok) else None

    def test_should_have_a_wired_meta_algorithm_entry_for_every_manifest_entry(self):
        """Every algorithm DescribeMain lists as registered must be wired=True here too."""
        manifest = self._manifest()
        if manifest is None:
            pytest.skip(f"Evolver jar not found at {jar_path()}")

        # Arrange
        manifest_names = {a["name"] for a in manifest["metaAlgorithms"]}
        wired_names = {a.name for a in META_ALGORITHMS if a.wired_into_cli_runner}

        # Assert
        assert manifest_names == wired_names

    def test_should_match_the_manifest_tree_support_of_every_wired_meta_algorithm(self):
        """supports_tree must agree with each manifest entry's supportsTree."""
        manifest = self._manifest()
        if manifest is None:
            pytest.skip(f"Evolver jar not found at {jar_path()}")

        # Arrange
        manifest_tree = {a["name"]: a["supportsTree"] for a in manifest["metaAlgorithms"]}
        catalogue_tree = {
            a.name: a.supports_tree for a in META_ALGORITHMS if a.wired_into_cli_runner
        }

        # Assert
        assert catalogue_tree == manifest_tree

    def test_should_have_a_runnable_base_algorithm_entry_for_every_manifest_entry(self):
        """Every algorithm DescribeMain lists as registered must be runnable_today=True here too."""
        manifest = self._manifest()
        if manifest is None:
            pytest.skip(f"Evolver jar not found at {jar_path()}")

        # Arrange
        manifest_names = {a["name"] for a in manifest["baseAlgorithms"]}
        runnable_names = {a.registry_name for a in BASE_ALGORITHMS if a.runnable_today}

        # Assert
        assert manifest_names == runnable_names

    def test_should_match_the_manifest_extra_config_keys_of_every_runnable_base_algorithm(self):
        """required_extra_config_keys must agree with each manifest's requiredExtraConfigKeys."""
        manifest = self._manifest()
        if manifest is None:
            pytest.skip(f"Evolver jar not found at {jar_path()}")

        # Arrange
        manifest_keys = {
            a["name"]: set(a["requiredExtraConfigKeys"]) for a in manifest["baseAlgorithms"]
        }
        catalogue_keys = {
            a.registry_name: set(a.required_extra_config_keys)
            for a in BASE_ALGORITHMS
            if a.runnable_today
        }

        # Assert
        assert catalogue_keys == manifest_keys

    def test_should_have_an_indicator_entry_for_every_manifest_indicator(self):
        """QUALITY_INDICATORS must list exactly what IndicatorRegistry registers."""
        manifest = self._manifest()
        if manifest is None:
            pytest.skip(f"Evolver jar not found at {jar_path()}")

        # Arrange
        catalogue_names = {indicator.registry_name for indicator in QUALITY_INDICATORS}

        # Assert
        assert catalogue_names == set(manifest["indicators"])
