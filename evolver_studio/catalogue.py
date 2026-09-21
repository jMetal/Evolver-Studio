"""Provisional catalogue of Evolver's base and meta-optimization algorithms.

Evolver's org.uma.evolver.cli.training.DescribeMain now exposes a machine-readable manifest of
what BaseAlgorithmRegistry/MetaAlgorithmRegistry actually register (see
evolver_client.describe() and Evolver's docs/proposals/cli-describe-manifest.md) — the
`runnable_today`/`wired_into_cli_runner` flags below should match it. What DescribeMain does
*not* cover is the broader, browsable-but-unregistered set this module also documents (e.g.
SMS-EMOA, RDE-MOEA as base algorithms; Async Genetic Algorithm as a meta-optimizer): those exist
as org.uma.evolver.algorithm.*/org.uma.evolver.meta.builder.* Java classes, usable from
org.uma.evolver.example.*, but never registered for cli.training — there is no registry to
introspect for them, so this module still mirrors them by hand and must be kept in sync manually
if Evolver's algorithm set changes there — see tests/test_catalogue.py for a check against the
real checkout.

`runnable_today`/`wired_into_cli_runner` distinguish "Evolver-Studio can browse this algorithm's
parameter space" (true for everything here — it's just reading a YAML file) from "Evolver-Studio can
actually launch a training run with it" (true only where org.uma.evolver.cli.training already
supports it: NSGA-II/MOEA-D as base algorithms; NSGA-II/SPEA2/SMPSO/AsyncNSGA-II/RandomSearch as
flat-encoding meta-optimizers, only NSGA-II for tree).
"""

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class BaseAlgorithm:
    """A base-level algorithm Evolver can tune, and its parameter space per encoding.

    Attributes:
        name: The algorithm's display name.
        encodings: Encoding name to its parameter space YAML filename, under
            Evolver's src/main/resources/parameterSpaces/.
        runnable_today: Whether org.uma.evolver.cli.training.BaseAlgorithmRegistry
            can currently resolve this algorithm to actually launch a run.
        registry_name: The exact string BaseAlgorithmRegistry.resolve() expects
            as its algorithmName argument, when it differs from `name` (e.g.
            "MOEAD", not "MOEA/D"). None when `runnable_today` is False.
        runnable_encoding: Which encoding key in `encodings` is the one
            BaseAlgorithmRegistry actually builds today — it hardcodes the
            Double-encoded Java class regardless of which parameter space
            file is passed in, so Binary/Permutation YAML files are only
            browsable, not launchable, even though the algorithm itself is
            `runnable_today`. None when `runnable_today` is False.
    """

    name: str
    encodings: dict[str, str]
    runnable_today: bool
    registry_name: str | None = None
    runnable_encoding: str | None = None


@dataclass(slots=True, frozen=True)
class MetaAlgorithm:
    """A meta-optimizer algorithm Evolver can use to search a base algorithm's parameter space.

    Attributes:
        name: The algorithm's name.
        supports_flat: Whether it can be used with the flat (array) encoding.
        supports_tree: Whether it can be used with the tree (grammar derivation)
            encoding. False for algorithms tied to a continuous solution
            representation (e.g. SMPSO's particle velocity).
        flat_parameters: Its configurable parameters under flat encoding, as a
            flat name list — fallback display for algorithms with no real
            ParameterSpace file (`operator_parameter_space_file` is None), since
            their operators are hardcoded Java, not data to read structure from.
        tree_parameters: Its configurable parameters under tree encoding, empty
            if `supports_tree` is False. Always a flat list: the tree encoding
            has no meta-level YAML at all (its two operators are fixed, only
            their scalar hyperparameters are configurable).
        wired_into_cli_runner: Whether org.uma.evolver.cli.training.MetaAlgorithmRegistry
            can currently use this as the meta-optimizer.
        example_config_file: Filename of a ready-to-use metaSearch configuration
            under src/main/resources/metaOptimizerConfigurations/, whose
            operator flags seed this algorithm's launch-form editor. None when
            not `wired_into_cli_runner` (nothing to launch).
        operator_parameter_space_file: Filename of the real ParameterSpace YAML
            backing this algorithm's flat-encoding operator catalogue, under
            src/main/resources/parameterSpaces/ (same format/parser as a base
            algorithm's own `yamlParameterSpaceFile`) — mirrors Evolver's
            MetaAlgorithmRegistry.MetaAlgorithmDescriptor.operatorParameterSpaceFile.
            None when the algorithm hardcodes its operators in Java instead
            (SPEA2, SMPSO); `flat_parameters` is the fallback for those.
    """

    name: str
    supports_flat: bool
    supports_tree: bool
    flat_parameters: tuple[str, ...]
    tree_parameters: tuple[str, ...]
    wired_into_cli_runner: bool
    example_config_file: str | None = None
    operator_parameter_space_file: str | None = None


BASE_ALGORITHMS: tuple[BaseAlgorithm, ...] = (
    # org.uma.evolver.algorithm.nsgaii.{Double,Binary,Permutation}NSGAII
    BaseAlgorithm(
        name="NSGA-II",
        encodings={
            "Double": "NSGAIIDouble.yaml",
            "Binary": "NSGAIIBinary.yaml",
            "Permutation": "NSGAIIPermutation.yaml",
        },
        runnable_today=True,
        registry_name="NSGA-II",
        runnable_encoding="Double",
    ),
    # org.uma.evolver.algorithm.moead.{Double,Binary,Permutation}MOEAD
    BaseAlgorithm(
        name="MOEA/D",
        encodings={
            "Double": "MOEADDouble.yaml",
            "Binary": "MOEADBinary.yaml",
            "Permutation": "MOEADPermutation.yaml",
        },
        runnable_today=True,
        registry_name="MOEAD",
        runnable_encoding="Double",
    ),
    # org.uma.evolver.algorithm.smsemoa.{Double,Binary,Permutation}SMSEMOA
    BaseAlgorithm(
        name="SMS-EMOA",
        encodings={
            "Double": "SMSEMOADouble.yaml",
            "Binary": "SMSEMOABinary.yaml",
            "Permutation": "SMSEMOAPermutation.yaml",
        },
        runnable_today=False,
    ),
    # org.uma.evolver.algorithm.rdemoea.{Double,Permutation}RDEMOEA (no Binary variant)
    BaseAlgorithm(
        name="RDE-MOEA",
        encodings={"Double": "RDEMOEADouble.yaml", "Permutation": "RDEMOEAPermutation.yaml"},
        runnable_today=False,
    ),
    # org.uma.evolver.algorithm.agemoea.DoubleAGEMOEA (Double only)
    BaseAlgorithm(
        name="AGE-MOEA", encodings={"Double": "AGEMOEADouble.yaml"}, runnable_today=False
    ),
    # org.uma.evolver.algorithm.rvea.DoubleRVEA (Double only)
    BaseAlgorithm(name="RVEA", encodings={"Double": "RVEADouble.yaml"}, runnable_today=False),
    # org.uma.evolver.algorithm.mopso.BaseMOPSO (Double only, particle swarm)
    BaseAlgorithm(name="MOPSO", encodings={"Double": "MOPSO.yaml"}, runnable_today=False),
)

# Every other file under parameterSpaces/ as of this writing, explicitly triaged as NOT a base
# algorithm's own parameter space — so a drift-detection test (see tests/test_catalogue.py) can flag
# any *new* file it doesn't recognize, instead of silently ignoring it or false-alarming on these.
KNOWN_NON_ALGORITHM_PARAMETER_SPACE_FILES = frozenset(
    {
        # Meta-level parameter spaces (the search space for the *meta*-optimizer's own operators,
        # passed as metaYamlParameterSpaceFile — smaller companions to a same-named file above).
        "NSGAIIDoubleReduced.yaml",
        "MOEADDoubleReduced.yaml",
        "RDEMOEADoubleReduced.yaml",
        "SMSEMOADoubleReduced.yaml",
        "MOPSOReduced.yaml",
        # Internal operator catalogues for the meta-optimizer itself (cli.training's
        # MetaAlgorithmRegistry), not a base-level algorithm's own parameter space — not
        # user-facing, hardcoded per registered meta-algorithm (NSGA-II/AsyncNSGA-II).
        "NSGAIIMetaDouble.yaml",
        "AsyncNSGAIIMetaDouble.yaml",
        # Orphaned: no Java class under org.uma.evolver.algorithm implements SSMOEA at all.
        "SSMOEADouble.yaml",
        # irace's own text format (see org.uma.evolver.irace.generator), not Evolver's YAML schema.
        "NSGAIIDouble.irace",
        "MOEADouble.irace",
    }
)

_FLAT_NSGAII_PARAMETERS = (
    "populationSize",
    "maxEvaluations",
    "numberOfCores",
    # The rest come from MetaAlgorithmRegistry's internal NSGAIIMetaDouble.yaml catalogue
    # (algorithmResult/createInitialSolutions/variation are fixed, not user-configurable):
    "offspringPopulationSize",
    "crossover",
    "mutation",
)
_TREE_NSGAII_PARAMETERS = (
    "metaPopulationSize",
    "metaOffspringSize",
    "numberOfCores",
    "metaMaxEvaluations",
    "crossoverProbability",
    "mutationProbability",
    "mutationDistributionIndex",
)

META_ALGORITHMS: tuple[MetaAlgorithm, ...] = (
    # org.uma.evolver.cli.training.MetaAlgorithmRegistry ("NSGA-II") — built on
    # DoubleNSGAII, wired for both flat (resolveFlat) and tree (validateTreeAlgorithm) encodings;
    # renamed from Evolver's own "NSGA-II" since every NSGA-II-shaped meta-optimizer evaluates in
    # parallel (MultiThreadedEvaluation).
    MetaAlgorithm(
        name="NSGA-II",
        supports_flat=True,
        supports_tree=True,
        flat_parameters=_FLAT_NSGAII_PARAMETERS,
        tree_parameters=_TREE_NSGAII_PARAMETERS,
        wired_into_cli_runner=True,
        example_config_file="MetaNSGAIIFlatConfiguration.yaml",
        operator_parameter_space_file="NSGAIIMetaDouble.yaml",
    ),
    # org.uma.evolver.cli.training.MetaAlgorithmRegistry ("SPEA2") — built via MetaSPEA2Builder,
    # which hardcodes its own operators (SBX, polynomial mutation, KNN density estimator,
    # tournament selection); only offspringPopulationSize/mutationProbabilityFactor are
    # configurable, both optional.
    MetaAlgorithm(
        name="SPEA2",
        supports_flat=True,
        supports_tree=False,
        flat_parameters=(
            "populationSize",
            "maxEvaluations",
            "numberOfCores",
            "offspringPopulationSize",
            "mutationProbabilityFactor",
        ),
        tree_parameters=(),
        wired_into_cli_runner=True,
        example_config_file="MetaSPEA2FlatConfiguration.yaml",
    ),
    # org.uma.evolver.cli.training.MetaAlgorithmRegistry ("SMPSO") — built via MetaSMPSOBuilder,
    # which exposes no operator catalogue at all (swarm size/evaluations/cores only); structurally
    # flat-only, since build() requires a DoubleProblem and tree encoding's DerivationTreeSolution
    # has no velocity/Double shape.
    MetaAlgorithm(
        name="SMPSO",
        supports_flat=True,
        supports_tree=False,
        flat_parameters=("swarmSize", "maxEvaluations", "numberOfCores"),
        tree_parameters=(),
        wired_into_cli_runner=True,
        example_config_file="MetaSMPSOFlatConfiguration.yaml",
    ),
    # org.uma.evolver.cli.training.MetaAlgorithmRegistry ("AsyncNSGA-II") — built via
    # MetaAsyncNSGAIIBuilder, which hardcodes its own selection/replacement; only its
    # crossover/mutation operators are configurable, via a much smaller parameter space.
    MetaAlgorithm(
        name="AsyncNSGA-II",
        supports_flat=True,
        supports_tree=False,
        flat_parameters=(
            "populationSize",
            "maxEvaluations",
            "numberOfCores",
            "crossover",
            "mutation",
        ),
        tree_parameters=(),
        wired_into_cli_runner=True,
        example_config_file="MetaAsyncNSGAIIFlatConfiguration.yaml",
        operator_parameter_space_file="AsyncNSGAIIMetaDouble.yaml",
    ),
    # org.uma.evolver.meta.builder.MetaAsyncGeneticAlgorithmBuilder
    MetaAlgorithm(
        name="Async Genetic Algorithm",
        supports_flat=True,
        supports_tree=False,
        flat_parameters=(
            "populationSize",
            "maxEvaluations",
            "numberOfCores",
            "mutationProbabilityFactor",
        ),
        tree_parameters=(),
        wired_into_cli_runner=False,
    ),
    # org.uma.evolver.cli.training.MetaAlgorithmRegistry ("RandomSearch") — built via
    # MetaRandomSearchBuilder, which has no population concept and exposes no operator catalogue
    # at all (it just samples uniformly at random); generic over the solution type, so it could
    # support tree too, but cli.training's tree pipeline (TrainingRunner.runTree(), via
    # validateTreeAlgorithm) only accepts NSGA-II today, same convention as SPEA2/SMPSO/
    # AsyncNSGA-II above (wired for flat only despite the underlying builder's own capability).
    MetaAlgorithm(
        name="RandomSearch",
        supports_flat=True,
        supports_tree=False,
        flat_parameters=("maxEvaluations", "numberOfCores"),
        tree_parameters=(),
        wired_into_cli_runner=True,
        example_config_file="MetaRandomSearchFlatConfiguration.yaml",
    ),
)
