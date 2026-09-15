"""Training request data and YAML serialization for Evolver's cli.runner."""

from dataclasses import dataclass

import yaml


@dataclass(slots=True, frozen=True)
class BaseLevelConfig:
    """What is being tuned and on which training set.

    Mirrors org.uma.evolver.cli.runner.BaseLevelConfig field for field.

    Attributes:
        algorithm_name: Base-level algorithm name (e.g. "NSGA-II").
        population_size: Base-level algorithm population size.
        number_of_independent_runs: Independent runs per training evaluation.
        yaml_parameter_space_file: Base-level algorithm parameter space YAML.
        extra_config: Algorithm-specific extra settings, or None.
        training_problem_names: Training problem names.
        training_reference_front_file_names: Reference front file per problem.
        training_evaluations: Base-level evaluation budget per problem.
        indicator_names: Quality indicator names.
        output_directory: Where results are written.
    """

    algorithm_name: str
    population_size: int
    number_of_independent_runs: int
    yaml_parameter_space_file: str
    extra_config: dict[str, str] | None
    training_problem_names: list[str]
    training_reference_front_file_names: list[str]
    training_evaluations: list[int]
    indicator_names: list[str]
    output_directory: str

    def to_dict(self) -> dict:
        """Build the mapping expected under the request's `baseLevel` key.

        Returns:
            A dict with the exact keys TrainingRequestYamlLoader reads.
        """
        return {
            "algorithmName": self.algorithm_name,
            "populationSize": self.population_size,
            "numberOfIndependentRuns": self.number_of_independent_runs,
            "yamlParameterSpaceFile": self.yaml_parameter_space_file,
            "extraConfig": self.extra_config,
            "trainingProblemNames": self.training_problem_names,
            "trainingReferenceFrontFileNames": self.training_reference_front_file_names,
            "trainingEvaluations": self.training_evaluations,
            "indicatorNames": self.indicator_names,
            "outputDirectory": self.output_directory,
        }


@dataclass(slots=True, frozen=True)
class FlatMetaSearchConfig:
    """How the meta-optimizer searches, for the flat (YAML) encoding.

    Mirrors org.uma.evolver.cli.runner.FlatMetaSearchConfig field for field.

    Attributes:
        meta_max_evaluations: Meta-level evaluation budget.
        meta_population_size: Meta-level population size.
        number_of_cores: Cores used to parallelize base-level runs.
        mutation_probability_factor: Meta-level mutation probability factor.
        meta_yaml_parameter_space_file: Meta-level algorithm parameter space YAML.
    """

    meta_max_evaluations: int
    meta_population_size: int
    number_of_cores: int
    mutation_probability_factor: float
    meta_yaml_parameter_space_file: str

    def to_dict(self) -> dict:
        """Build the mapping expected under the request's `metaSearch` key.

        Returns:
            A dict with the exact keys TrainingRequestYamlLoader reads,
            including the `encoding: "flat"` discriminator.
        """
        return {
            "encoding": "flat",
            "metaMaxEvaluations": self.meta_max_evaluations,
            "metaPopulationSize": self.meta_population_size,
            "numberOfCores": self.number_of_cores,
            "mutationProbabilityFactor": self.mutation_probability_factor,
            "metaYamlParameterSpaceFile": self.meta_yaml_parameter_space_file,
        }


def to_request_yaml(base_level: BaseLevelConfig, meta_search: FlatMetaSearchConfig) -> str:
    """Serialize a training request to the YAML TrainingRunnerMain expects.

    Args:
        base_level: What is being tuned and on which training set.
        meta_search: How the meta-optimizer searches (flat encoding).

    Returns:
        The YAML document text, with `baseLevel` and `metaSearch` top-level keys.
    """
    request = {"baseLevel": base_level.to_dict(), "metaSearch": meta_search.to_dict()}
    return yaml.safe_dump(request, sort_keys=False)
