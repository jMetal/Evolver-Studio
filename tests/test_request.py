"""Tests for training request YAML serialization."""

import yaml

from evolver_studio.request import BaseLevelConfig, FlatMetaSearchConfig, to_request_yaml


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
        "output_directory": "results/nsgaii/ZDT4",
    }
    defaults.update(overrides)
    return BaseLevelConfig(**defaults)


def _meta_search(**overrides) -> FlatMetaSearchConfig:
    defaults = {
        "meta_max_evaluations": 2000,
        "meta_population_size": 100,
        "number_of_cores": 8,
        "mutation_probability_factor": 1.5,
        "meta_yaml_parameter_space_file": "NSGAIIDoubleReduced.yaml",
    }
    defaults.update(overrides)
    return FlatMetaSearchConfig(**defaults)


class TestToRequestYaml:
    def test_should_serialize_flat_metasearch_with_encoding_key(self):
        """The metaSearch section must carry the flat encoding discriminator."""
        # Arrange
        base_level = _base_level()
        meta_search = _meta_search()

        # Act
        parsed = yaml.safe_load(to_request_yaml(base_level, meta_search))

        # Assert
        assert parsed["metaSearch"]["encoding"] == "flat"

    def test_should_round_trip_all_base_level_fields(self):
        """Every BaseLevelConfig field must map to its exact expected YAML key."""
        # Arrange
        base_level = _base_level()
        meta_search = _meta_search()

        # Act
        parsed = yaml.safe_load(to_request_yaml(base_level, meta_search))

        # Assert
        assert parsed["baseLevel"] == {
            "algorithmName": "NSGA-II",
            "populationSize": 100,
            "numberOfIndependentRuns": 1,
            "yamlParameterSpaceFile": "NSGAIIDouble.yaml",
            "extraConfig": None,
            "trainingProblemNames": ["ZDT4"],
            "trainingReferenceFrontFileNames": ["resources/referenceFronts/ZDT4.csv"],
            "trainingEvaluations": [12000],
            "indicatorNames": ["Epsilon", "NormalizedHypervolume"],
            "outputDirectory": "results/nsgaii/ZDT4",
        }

    def test_should_serialize_null_extra_config_as_yaml_null(self):
        """extra_config=None must serialize as YAML null, not be omitted."""
        # Arrange
        base_level = _base_level(extra_config=None)
        meta_search = _meta_search()

        # Act
        parsed = yaml.safe_load(to_request_yaml(base_level, meta_search))

        # Assert
        assert parsed["baseLevel"]["extraConfig"] is None

    def test_should_serialize_extra_config_when_present(self):
        """extra_config, when given, must serialize as a plain string map."""
        # Arrange
        extra = {"weightVectorFilesDirectory": "resources/weightVectors"}
        base_level = _base_level(extra_config=extra)
        meta_search = _meta_search()

        # Act
        parsed = yaml.safe_load(to_request_yaml(base_level, meta_search))

        # Assert
        assert parsed["baseLevel"]["extraConfig"] == extra
