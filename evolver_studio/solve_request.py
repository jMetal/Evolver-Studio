"""A solve request: Evolver's cli.solving request, and its YAML serialization.

Mirrors org.uma.evolver.cli.solving.SolveRequest field for field (see Evolver's
docs/proposals/cli-solving.md). Unlike a training request it is a single self-contained file.
"""

from dataclasses import dataclass

import yaml

from evolver_studio.problem_catalogue import ArgumentValue, problem_spec


@dataclass(slots=True, frozen=True)
class SolveRequest:
    """A configurable algorithm, with a configuration, run on a problem one or more times.

    Attributes:
        algorithm_name: The algorithm's registry name (e.g. "NSGA-II", "MOEAD").
        encoding: The solution encoding (e.g. "Double", "Permutation").
        population_size: The population size.
        yaml_parameter_space_file: The parameter space the configuration is parsed against.
        extra_config: Algorithm-specific extra settings (e.g. MOEA/D's
            weightVectorFilesDirectory), or None.
        configuration: The configuration string, "--parameter value ...".
        problem: The problem's name.
        reference_front_file_name: The problem's reference front, or None; required to compute
            indicators.
        max_evaluations: The evaluation budget of each run.
        number_of_independent_runs: How many independent runs.
        seed: The seed of the first run (run i uses seed + i - 1), or None to draw one at random.
        indicator_names: The quality indicators computed for each run; empty for none.
        status_frequency: Every how many evaluations Evolver updates the status while a run is in
            progress, or None to update it only when a run ends (Evolver 2.3 or later; older ones
            ignore it). The more often, the slower the run.
        front_frequency: Every how many evaluations Evolver writes the current front while a run
            is in progress, or None to write none (Evolver 2.3 or later). The more often, the
            slower the run, and more so than the status.
        write_population: Whether that file holds the whole population instead of only the
            non-dominated solutions; needs `front_frequency`.
        output_directory: Where the results are written.
        problem_arguments: The problem's constructor arguments, all of them in order, or empty to
            build it with none.
    """

    algorithm_name: str
    encoding: str
    population_size: int
    yaml_parameter_space_file: str
    extra_config: dict[str, str] | None
    configuration: str
    problem: str
    reference_front_file_name: str | None
    max_evaluations: int
    number_of_independent_runs: int
    seed: int | None
    indicator_names: list[str]
    status_frequency: int | None
    front_frequency: int | None
    write_population: bool
    output_directory: str
    problem_arguments: tuple[ArgumentValue, ...] = ()

    def validation_errors(self) -> list[str]:
        """Check what Evolver would reject, to tell the user before launching.

        Returns:
            One message per problem found; empty when the request is valid.
        """
        checks = [
            (not self.problem, "Choose a problem."),
            (not self.algorithm_name, "Choose an algorithm."),
            (not self.configuration.strip(), "The configuration is empty."),
            (
                bool(self.indicator_names) and not self.reference_front_file_name,
                "The indicators need a reference front: choose one, or no indicators.",
            ),
            (self.population_size < 1, "The population size must be positive."),
            (self.max_evaluations < 1, "The maximum number of evaluations must be positive."),
            (self.number_of_independent_runs < 1, "At least one run is needed."),
            (
                self.status_frequency is not None and self.status_frequency < 1,
                "The progress must be updated every one evaluation or more.",
            ),
            (
                self.front_frequency is not None and self.front_frequency < 1,
                "The front must be updated every one evaluation or more.",
            ),
            (
                self.write_population and self.front_frequency is None,
                "Showing the whole population needs the live front.",
            ),
        ]
        return [message for failed, message in checks if failed]


def solve_request_to_yaml(request: SolveRequest) -> str:
    """Serialize a solve request to the YAML SolveRunnerMain reads.

    Args:
        request: The request.

    Returns:
        The YAML document text. `seed`, `referenceFrontFileName`, `statusFrequency` and
        `frontFrequency` are left out when unset, so Evolver draws a seed, computes no indicators,
        updates the status only when a run ends and writes no front meanwhile.
    """
    data: dict = {
        "algorithmName": request.algorithm_name,
        "encoding": request.encoding,
        "populationSize": request.population_size,
        "yamlParameterSpaceFile": request.yaml_parameter_space_file,
        "extraConfig": request.extra_config or {},
        "configuration": request.configuration,
        "problem": problem_spec(request.problem, request.problem_arguments),
    }
    if request.reference_front_file_name:
        data["referenceFrontFileName"] = request.reference_front_file_name
    data["maxEvaluations"] = request.max_evaluations
    data["numberOfIndependentRuns"] = request.number_of_independent_runs
    if request.seed is not None:
        data["seed"] = request.seed
    data["indicatorNames"] = request.indicator_names
    if request.status_frequency is not None:
        data["statusFrequency"] = request.status_frequency
    if request.front_frequency is not None:
        data["frontFrequency"] = request.front_frequency
        if request.write_population:
            data["writePopulation"] = True
    data["outputDirectory"] = request.output_directory
    return yaml.safe_dump(data, sort_keys=False)
