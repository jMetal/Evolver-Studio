"""Training request data and YAML serialization for Evolver's cli.training.

A full request is three independent files (see cli.training in Evolver's
docs/utilities/cli_tools.rst): request.yaml itself (baseLevel/metaSearch as file
names, plus outputDirectory and the optional
writeFrequency/statusFrequency/frontPlotFrequency, all specific to *this* run),
the baseLevel file (what is tuned, reusable), and the metaSearch file (how the
meta-optimizer searches, reusable).
"""

from dataclasses import dataclass

import yaml

# Scalar keys MetaOptimizerConfigurationReader consumes directly, for the flat and the tree
# encoding; every other key in a metaSearch file is an operator flag, passed straight through to
# the algorithm's own operator parameter space (see MetaSearchConfig.operator_flags).
META_SEARCH_SCALAR_KEYS = frozenset(
    {
        "algorithm",
        "encoding",
        "metaMaxEvaluations",
        "metaMaxComputingTimeMinutes",
        "metaPopulationSize",
        "numberOfCores",
    }
)


# The meta-optimizer's population size when the request leaves it to Evolver
# (MetaAlgorithmRegistry.DEFAULT_POPULATION_SIZE).
DEFAULT_META_POPULATION_SIZE = 50


def checkpoint_frequency_error(
    frequency: int, meta_population_size: int = DEFAULT_META_POPULATION_SIZE
) -> str | None:
    """Say why a checkpoint frequency would leave a training run without checkpoints.

    Evolver writes its result files and its status when the evaluations are a multiple of the
    frequency, and the generational meta-optimizers only publish them in multiples of their
    population size: any other frequency writes nothing until the very end.

    Args:
        frequency: The `writeFrequency` and `statusFrequency` of the request, in meta-evaluations.
        meta_population_size: The meta-optimizer's population size.

    Returns:
        The message, or None when the frequency is fine.
    """
    if frequency >= meta_population_size and frequency % meta_population_size == 0:
        return None
    return (
        f"Update every {meta_population_size}, {2 * meta_population_size}, ... evaluations: "
        f"Evolver writes a checkpoint only when the evaluations are a multiple of the frequency, "
        f"and the meta-optimizer completes them in multiples of its population size "
        f"({meta_population_size}); {frequency} would give no checkpoint until the end."
    )


@dataclass(slots=True, frozen=True)
class BaseLevelConfig:
    """What is being tuned and on which training set.

    Mirrors org.uma.evolver.cli.training.BaseLevelConfig field for field.
    Reusable across many requests, so it carries no output-directory or
    other per-run detail — see TrainingRequest for those.

    Attributes:
        algorithm_name: Base-level algorithm name (e.g. "NSGA-II").
        population_size: Base-level algorithm population size.
        number_of_independent_runs: Independent runs per training evaluation.
        yaml_parameter_space_file: Base-level algorithm parameter space YAML.
        extra_config: Algorithm-specific extra settings, or None.
        training_problem_names: Training problems: a name, or a `{class, args}` map for a
            problem built with arguments.
        training_reference_front_file_names: Reference front file per problem.
        training_evaluations: Base-level evaluation budget per problem.
        indicator_names: Quality indicator names.
        encoding: The jMetal solution encoding the base-level algorithm is
            built for (e.g. "Double", "Permutation"), resolved together with
            `algorithm_name` by BaseAlgorithmRegistry. Defaults to "Double",
            matching BaseLevelConfigurationReader's own default when the
            field is absent from a hand-written YAML file.
    """

    algorithm_name: str
    population_size: int
    number_of_independent_runs: int
    yaml_parameter_space_file: str
    extra_config: dict[str, str] | None
    training_problem_names: list[str | dict]
    training_reference_front_file_names: list[str]
    training_evaluations: list[int]
    indicator_names: list[str]
    encoding: str = "Double"

    def to_dict(self) -> dict:
        """Build the mapping BaseLevelConfigurationReader expects.

        Returns:
            A dict with the exact keys BaseLevelConfigurationReader reads.
        """
        return {
            "algorithmName": self.algorithm_name,
            "encoding": self.encoding,
            "populationSize": self.population_size,
            "numberOfIndependentRuns": self.number_of_independent_runs,
            "yamlParameterSpaceFile": self.yaml_parameter_space_file,
            "extraConfig": self.extra_config,
            "trainingProblemNames": self.training_problem_names,
            "trainingReferenceFrontFileNames": self.training_reference_front_file_names,
            "trainingEvaluations": self.training_evaluations,
            "indicatorNames": self.indicator_names,
        }


@dataclass(slots=True, frozen=True)
class MetaSearchConfig:
    """How the meta-optimizer searches, for the flat [0,1]^n or the tree encoding.

    Mirrors org.uma.evolver.cli.training.FlatMetaSearchConfig and TreeMetaSearchConfig, which
    have the same fields: the encoding is the `encoding` key of the file. Reusable across many
    requests.

    Attributes:
        algorithm: The meta-optimizer algorithm name, resolved via
            MetaAlgorithmRegistry (e.g. "NSGA-II", "SPEA2",
            "AsyncNSGA-II", "SMPSO").
        meta_max_evaluations: Meta-level evaluation budget; ignored (and left out of the file)
            when `meta_max_computing_time_minutes` is given: the two limits are exclusive.
        meta_population_size: Meta-level population size, or None to use
            MetaAlgorithmRegistry's own default.
        number_of_cores: Cores used to parallelize base-level runs.
        operator_flags: The meta-optimizer's own operator configuration
            (crossover, mutation, ...), as plain key/value pairs — every
            algorithm accepts a different set (SMPSO accepts none), and the tree encoding
            has others than the flat one.
        encoding: "flat" or "tree".
        meta_max_computing_time_minutes: Stop the meta-optimizer after this many minutes
            (decimals allowed) instead of after `meta_max_evaluations`, or None for the latter.
    """

    algorithm: str
    meta_max_evaluations: int
    meta_population_size: int | None
    number_of_cores: int
    operator_flags: dict[str, object]
    encoding: str = "flat"
    meta_max_computing_time_minutes: float | None = None

    def to_dict(self) -> dict:
        """Build the mapping MetaOptimizerConfigurationReader expects.

        Returns:
            A dict with the scalar keys, the `encoding` discriminator ("flat" or
            "tree"), and every operator flag as a top-level key.
        """
        data = {
            "algorithm": self.algorithm,
            "encoding": self.encoding,
        }
        if self.meta_max_computing_time_minutes is not None:
            data["metaMaxComputingTimeMinutes"] = self.meta_max_computing_time_minutes
        else:
            data["metaMaxEvaluations"] = self.meta_max_evaluations
        data["numberOfCores"] = self.number_of_cores
        if self.meta_population_size is not None:
            data["metaPopulationSize"] = self.meta_population_size
        data.update(self.operator_flags)
        return data


def base_level_to_yaml(base_level: BaseLevelConfig) -> str:
    """Serialize a base-level config to a reusable baseLevel configuration file.

    Args:
        base_level: What is being tuned and on which training set.

    Returns:
        The YAML document text, readable by BaseLevelConfigurationReader.
    """
    return yaml.safe_dump(base_level.to_dict(), sort_keys=False)


def meta_search_to_yaml(meta_search: MetaSearchConfig) -> str:
    """Serialize a meta-search config to a reusable metaSearch configuration file.

    Args:
        meta_search: How the meta-optimizer searches (flat or tree encoding).

    Returns:
        The YAML document text, readable by MetaOptimizerConfigurationReader.
    """
    return yaml.safe_dump(meta_search.to_dict(), sort_keys=False)


def request_to_yaml(
    base_level_file: str,
    meta_search_file: str,
    output_directory: str,
    write_frequency: int | None = None,
    status_frequency: int | None = None,
) -> str:
    """Serialize the top-level request.yaml TrainingRunnerMain expects.

    `frontPlotFrequency` is deliberately never set: Evolver-Studio runs
    headless, and its absence is what keeps a Swing window from popping up.

    Args:
        base_level_file: Path to the written baseLevel configuration file.
        meta_search_file: Path to the written metaSearch configuration file.
        output_directory: Where Evolver writes this run's results.
        write_frequency: How often (in evaluations) CONFIGURATIONS.csv/
            INDICATORS.csv are written, or None for TrainingRunnerMain's
            own default (100).
        status_frequency: How often (in evaluations) status.yaml is
            updated, or None for TrainingRunnerMain's own default (100).

    Returns:
        The YAML document text, with baseLevel/metaSearch as file-name
        references and outputDirectory at the top level.
    """
    request: dict = {
        "baseLevel": base_level_file,
        "metaSearch": meta_search_file,
        "outputDirectory": output_directory,
    }
    if write_frequency is not None:
        request["writeFrequency"] = write_frequency
    if status_frequency is not None:
        request["statusFrequency"] = status_frequency
    return yaml.safe_dump(request, sort_keys=False)


def parse_operator_flags_yaml(text: str) -> dict[str, object]:
    """Parse a flat YAML mapping of meta-optimizer operator flags.

    Args:
        text: YAML text, expected to be a flat mapping (e.g. `crossover:
            SBX`) or empty.

    Returns:
        The parsed mapping, or an empty dict for blank/null input.

    Raises:
        ValueError: If the text does not parse to a mapping.
    """
    parsed = yaml.safe_load(text)
    if parsed is None:
        return {}
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected a flat YAML mapping, got: {type(parsed).__name__}")
    return parsed
