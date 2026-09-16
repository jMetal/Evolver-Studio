"""Provisional catalogue of Evolver's base and meta-optimization algorithms.

No Java registry lists these today — org.uma.evolver.cli.runner.BaseAlgorithmRegistry only resolves
"NSGA-II"/"MOEAD" (its own javadoc calls this "prototype scope"), and the algorithm+encoding to YAML
filename mapping is a pure, unenforced naming convention scattered across org.uma.evolver.example.*
(<Algorithm><Encoding>.yaml). This module mirrors what the underlying org.uma.evolver.algorithm.*
and org.uma.evolver.meta.builder.* Java classes actually provide, so it must be kept in sync by hand
if Evolver's algorithm set changes — see tests/test_catalogue.py for a check against the real
checkout.

`runnable_today`/`wired_into_cli_runner` distinguish "Evolver-Studio can browse this algorithm's
parameter space" (true for everything here — it's just reading a YAML file) from "Evolver-Studio can
actually launch a training run with it" (true only where org.uma.evolver.cli.runner already supports
it: NSGA-II/MOEA-D as base algorithms, NSGA-II as meta-optimizer).
"""

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class BaseAlgorithm:
    """A base-level algorithm Evolver can tune, and its parameter space per encoding.

    Attributes:
        name: The algorithm's name.
        encodings: Encoding name to its parameter space YAML filename, under
            Evolver's src/main/resources/parameterSpaces/.
        runnable_today: Whether org.uma.evolver.cli.runner.BaseAlgorithmRegistry
            can currently resolve this algorithm to actually launch a run.
    """

    name: str
    encodings: dict[str, str]
    runnable_today: bool


@dataclass(slots=True, frozen=True)
class MetaAlgorithm:
    """A meta-optimizer algorithm Evolver can use to search a base algorithm's parameter space.

    Attributes:
        name: The algorithm's name.
        supports_flat: Whether it can be used with the flat (array) encoding.
        supports_tree: Whether it can be used with the tree (grammar derivation)
            encoding. False for algorithms tied to a continuous solution
            representation (e.g. SMPSO's particle velocity).
        flat_parameters: Its configurable parameters under flat encoding.
        tree_parameters: Its configurable parameters under tree encoding, empty
            if `supports_tree` is False.
        wired_into_cli_runner: Whether org.uma.evolver.cli.runner.TrainingRunner
            can currently use this as the meta-optimizer.
    """

    name: str
    supports_flat: bool
    supports_tree: bool
    flat_parameters: tuple[str, ...]
    tree_parameters: tuple[str, ...]
    wired_into_cli_runner: bool


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
        # Orphaned: no Java class under org.uma.evolver.algorithm implements SSMOEA at all.
        "SSMOEADouble.yaml",
        # irace's own text format (see org.uma.evolver.irace.generator), not Evolver's YAML schema.
        "NSGAIIDouble.irace",
        "MOEADouble.irace",
    }
)

_FLAT_NSGAII_PARAMETERS = (
    "populationSize",
    "offspringPopulationSize",  # supported by MetaNSGAIIBuilder, not yet exposed by cli.runner
    "maxEvaluations",
    "numberOfCores",
    "mutationProbabilityFactor",
    "metaYamlParameterSpaceFile",
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
    # org.uma.evolver.meta.builder.MetaNSGAIIBuilder — the only one wired into TrainingRunner today,
    # for both flat (runFlat) and tree (runTree's hand-assembled NSGA-II-shaped loop) encodings.
    MetaAlgorithm(
        name="NSGA-II",
        supports_flat=True,
        supports_tree=True,
        flat_parameters=_FLAT_NSGAII_PARAMETERS,
        tree_parameters=_TREE_NSGAII_PARAMETERS,
        wired_into_cli_runner=True,
    ),
    # org.uma.evolver.meta.builder.MetaSPEA2Builder — same shape as MetaNSGAIIBuilder, hardcodes its
    # own meta-level parameter space (RDEMOEADouble.yaml) rather than taking one as an argument.
    MetaAlgorithm(
        name="SPEA2",
        supports_flat=True,
        supports_tree=False,
        flat_parameters=(
            "populationSize",
            "offspringPopulationSize",
            "maxEvaluations",
            "numberOfCores",
            "mutationProbabilityFactor",
        ),
        tree_parameters=(),
        wired_into_cli_runner=False,
    ),
    # org.uma.evolver.meta.builder.MetaSMPSOBuilder — structurally flat-only: build() requires
    # a DoubleProblem, and tree encoding's DerivationTreeSolution has no velocity/Double shape.
    MetaAlgorithm(
        name="SMPSO",
        supports_flat=True,
        supports_tree=False,
        flat_parameters=("swarmSize", "maxEvaluations", "numberOfCores"),
        tree_parameters=(),
        wired_into_cli_runner=False,
    ),
    # org.uma.evolver.meta.builder.MetaAsyncNSGAIIBuilder
    MetaAlgorithm(
        name="Async NSGA-II",
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
    # org.uma.evolver.meta.builder.MetaRandomSearchBuilder<S> — generic over the solution type, so
    # the only one (besides NSGA-II) genuinely usable with either encoding.
    MetaAlgorithm(
        name="Random Search",
        supports_flat=True,
        supports_tree=True,
        flat_parameters=("maxEvaluations", "numberOfCores"),
        tree_parameters=("maxEvaluations", "numberOfCores"),
        wired_into_cli_runner=False,
    ),
)
