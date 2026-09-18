"""Tests for training request YAML serialization."""

import pytest
import yaml

from evolver_studio.request import (
    BaseLevelConfig,
    FlatMetaSearchConfig,
    base_level_to_yaml,
    flat_meta_search_to_yaml,
    parse_operator_flags_yaml,
    request_to_yaml,
)


def _base_level(**overrides) -> BaseLevelConfig:
    defaults = {
        "algorithm_name": "NSGA-II",
        "population_size": 100,
        "number_of_independent_runs": 1,
        "yaml_parameter_space_file": "NSGAIIDouble.yaml",
        "extra_config": None,
        "training_problem_names": ["ZDT4"],
        "training_reference_front_file_names": ["resources/referenceFronts/ZDT4.csv"],
        "training_evaluations": [12000],
        "indicator_names": ["Epsilon", "NormalizedHypervolume"],
    }
    defaults.update(overrides)
    return BaseLevelConfig(**defaults)


def _meta_search(**overrides) -> FlatMetaSearchConfig:
    defaults = {
        "algorithm": "ParallelNSGA-II",
        "meta_max_evaluations": 2000,
        "meta_population_size": 100,
        "number_of_cores": 8,
        "operator_flags": {"crossover": "SBX", "crossoverProbability": 0.9},
    }
    defaults.update(overrides)
    return FlatMetaSearchConfig(**defaults)


class TestBaseLevelToYaml:
    def test_should_round_trip_all_fields(self):
        """Every BaseLevelConfig field must map to its exact expected YAML key."""
        # Arrange
        base_level = _base_level()

        # Act
        parsed = yaml.safe_load(base_level_to_yaml(base_level))

        # Assert
        assert parsed == {
            "algorithmName": "NSGA-II",
            "populationSize": 100,
            "numberOfIndependentRuns": 1,
            "yamlParameterSpaceFile": "NSGAIIDouble.yaml",
            "extraConfig": None,
            "trainingProblemNames": ["ZDT4"],
            "trainingReferenceFrontFileNames": ["resources/referenceFronts/ZDT4.csv"],
            "trainingEvaluations": [12000],
            "indicatorNames": ["Epsilon", "NormalizedHypervolume"],
        }

    def test_should_serialize_null_extra_config_as_yaml_null(self):
        """extra_config=None must serialize as YAML null, not be omitted."""
        # Arrange
        base_level = _base_level(extra_config=None)

        # Act
        parsed = yaml.safe_load(base_level_to_yaml(base_level))

        # Assert
        assert parsed["extraConfig"] is None

    def test_should_serialize_extra_config_when_present(self):
        """extra_config, when given, must serialize as a plain string map."""
        # Arrange
        extra = {"weightVectorFilesDirectory": "resources/weightVectors"}
        base_level = _base_level(extra_config=extra)

        # Act
        parsed = yaml.safe_load(base_level_to_yaml(base_level))

        # Assert
        assert parsed["extraConfig"] == extra

    def test_should_not_carry_an_output_directory_key(self):
        """outputDirectory moved to the top-level request, not the (reusable) baseLevel file."""
        # Arrange
        base_level = _base_level()

        # Act
        parsed = yaml.safe_load(base_level_to_yaml(base_level))

        # Assert
        assert "outputDirectory" not in parsed


class TestFlatMetaSearchToYaml:
    def test_should_serialize_flat_encoding_discriminator(self):
        """The metaSearch file must carry the flat encoding discriminator."""
        # Arrange
        meta_search = _meta_search()

        # Act
        parsed = yaml.safe_load(flat_meta_search_to_yaml(meta_search))

        # Assert
        assert parsed["encoding"] == "flat"

    def test_should_serialize_algorithm_and_scalars(self):
        """algorithm/metaMaxEvaluations/metaPopulationSize/numberOfCores are top-level keys."""
        # Arrange
        meta_search = _meta_search()

        # Act
        parsed = yaml.safe_load(flat_meta_search_to_yaml(meta_search))

        # Assert
        assert parsed["algorithm"] == "ParallelNSGA-II"
        assert parsed["metaMaxEvaluations"] == 2000
        assert parsed["metaPopulationSize"] == 100
        assert parsed["numberOfCores"] == 8

    def test_should_flatten_operator_flags_as_top_level_keys(self):
        """Operator flags are not nested; each becomes its own top-level YAML key."""
        # Arrange
        meta_search = _meta_search(operator_flags={"crossover": "SBX", "mutation": "Polynomial"})

        # Act
        parsed = yaml.safe_load(flat_meta_search_to_yaml(meta_search))

        # Assert
        assert parsed["crossover"] == "SBX"
        assert parsed["mutation"] == "Polynomial"

    def test_should_omit_meta_population_size_when_none(self):
        """A None meta_population_size must be omitted, not written as YAML null."""
        # Arrange
        meta_search = _meta_search(meta_population_size=None)

        # Act
        parsed = yaml.safe_load(flat_meta_search_to_yaml(meta_search))

        # Assert
        assert "metaPopulationSize" not in parsed


class TestRequestToYaml:
    def test_should_reference_base_level_and_meta_search_by_file_name(self):
        """baseLevel/metaSearch are file-name references, not inline objects."""
        # Act
        parsed = yaml.safe_load(
            request_to_yaml("base_level.yaml", "meta_search.yaml", "results/run1")
        )

        # Assert
        assert parsed["baseLevel"] == "base_level.yaml"
        assert parsed["metaSearch"] == "meta_search.yaml"

    def test_should_put_output_directory_at_the_top_level(self):
        """outputDirectory is a request.yaml field, not nested under baseLevel."""
        # Act
        parsed = yaml.safe_load(
            request_to_yaml("base_level.yaml", "meta_search.yaml", "results/run1")
        )

        # Assert
        assert parsed["outputDirectory"] == "results/run1"

    def test_should_omit_write_and_status_frequency_by_default(self):
        """Leaving these unset lets TrainingRunnerMain apply its own default (100)."""
        # Act
        parsed = yaml.safe_load(
            request_to_yaml("base_level.yaml", "meta_search.yaml", "results/run1")
        )

        # Assert
        assert "writeFrequency" not in parsed
        assert "statusFrequency" not in parsed

    def test_should_include_write_and_status_frequency_when_given(self):
        """An explicit cadence must override TrainingRunnerMain's default."""
        # Act
        parsed = yaml.safe_load(
            request_to_yaml(
                "base_level.yaml",
                "meta_search.yaml",
                "results/run1",
                write_frequency=50,
                status_frequency=25,
            )
        )

        # Assert
        assert parsed["writeFrequency"] == 50
        assert parsed["statusFrequency"] == 25

    def test_should_never_set_front_plot_frequency(self):
        """Evolver-Studio runs headless; a live Swing plot must never be requested."""
        # Act
        parsed = yaml.safe_load(
            request_to_yaml("base_level.yaml", "meta_search.yaml", "results/run1")
        )

        # Assert
        assert "frontPlotFrequency" not in parsed


class TestParseOperatorFlagsYaml:
    def test_should_parse_a_flat_mapping(self):
        """The common case: a flat YAML mapping of operator flags."""
        # Act
        parsed = parse_operator_flags_yaml("crossover: SBX\ncrossoverProbability: 0.9\n")

        # Assert
        assert parsed == {"crossover": "SBX", "crossoverProbability": 0.9}

    def test_should_return_empty_dict_for_blank_text(self):
        """SMPSO accepts no operator flags at all; blank text must not be an error."""
        # Act
        parsed = parse_operator_flags_yaml("")

        # Assert
        assert parsed == {}

    def test_should_reject_a_non_mapping(self):
        """A YAML list/scalar is not a valid flat operator-flags mapping."""
        # Act & Assert
        with pytest.raises(ValueError, match="mapping"):
            parse_operator_flags_yaml("- SBX\n- Polynomial\n")
