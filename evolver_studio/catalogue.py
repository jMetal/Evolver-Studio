"""Provisional catalogue of Evolver's base and meta-optimization algorithms.

Evolver's org.uma.evolver.cli.training.DescribeMain now exposes a machine-readable manifest of
what BaseAlgorithmRegistry/MetaAlgorithmRegistry actually register (see
evolver_client.describe(), and Evolver's docs/utilities/cli_tools.rst) — the
`runnable_today`/`wired_into_cli_runner` flags below should match it. What DescribeMain does
*not* cover is the browsable-but-unregistered set this module also documents (e.g. MOPSO as a base
algorithm; Async Genetic Algorithm as a meta-optimizer): those exist as
org.uma.evolver.algorithm.*/org.uma.evolver.meta.{algorithm,builder}.* Java classes, usable from
org.uma.evolver.example.*, but never registered for cli.training — there is no registry to
introspect for them, so this module still mirrors them by hand and must be kept in sync manually
if Evolver's algorithm set changes there — see tests/test_catalogue.py for a check against the
parameter spaces packaged in Evolver's jar.

`runnable_today`/`wired_into_cli_runner` distinguish "Evolver-Studio can browse this algorithm's
parameter space" (true for everything here — it's just reading a YAML file) from "Evolver-Studio can
actually launch a training run with it" (true only where org.uma.evolver.cli.training already
supports it: every base algorithm but MOPSO (not registered); NSGA-II/AGE-MOEA/SPEA2/SMPSO/
AsyncNSGA-II/RandomSearch as flat-encoding meta-optimizers, NSGA-II/AGE-MOEA/AsyncNSGA-II/
RandomSearch for tree).
"""

from dataclasses import dataclass, field

from evolver_studio.evolver_client import EVOLVER_VERSION


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
        runnable_encodings: Which encoding keys in `encodings` BaseAlgorithmRegistry
            actually builds today (e.g. NSGA-II: both "Double" and "Permutation",
            each routing to a different Java class via BaseLevelConfig.encoding) —
            any other key in `encodings` is browsable only, not launchable, even
            though the algorithm itself is `runnable_today`. Empty when
            `runnable_today` is False.
        required_extra_config_keys: The extra configuration entries
            BaseAlgorithmRegistry needs to build it (its manifest's
            requiredExtraConfigKeys), e.g. "weightVectorFilesDirectory" for
            the decomposition-based ones. Empty when it needs none.
        default_configurations: Encoding name to the default configurations Evolver's jar ships
            for it, as (label, filename) pairs, under defaultConfigurations/ — the ones the Run
            algorithm page offers to start from. An encoding with none is missing (e.g. NSGA-II's
            Permutation).
    """

    name: str
    encodings: dict[str, str]
    runnable_today: bool
    registry_name: str | None = None
    runnable_encodings: tuple[str, ...] = ()
    required_extra_config_keys: tuple[str, ...] = ()
    default_configurations: dict[str, tuple[tuple[str, str], ...]] = field(default_factory=dict)


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
            if `supports_tree` is False — fallback display for algorithms with
            no `tree_operator_parameter_space_file` (e.g. RandomSearch, which
            has no operators at all).
        wired_into_cli_runner: Whether org.uma.evolver.cli.training.MetaAlgorithmRegistry
            can currently use this as the meta-optimizer.
        example_config_file: Filename of a ready-to-use metaSearch configuration
            under src/main/resources/metaOptimizerConfigurations/, whose
            operator flags seed this algorithm's launch-form editor. None when
            not `wired_into_cli_runner` (nothing to launch).
        tree_example_config_file: The same for the tree encoding, or None when the algorithm does
            not support it (`supports_tree` is False).
        uses_population: Whether the algorithm has a population whose size is chosen
            (`metaPopulationSize`); False for Random Search, which has none.
        fixed_operators_note: What to tell the user about the operators the algorithm hardcodes
            (and so cannot be chosen), or None when it has a catalogue of them.
        flat_only_reason: Why the algorithm does not support the tree encoding, or None when it
            does.
        operator_parameter_space_file: Filename of the real ParameterSpace YAML
            backing this algorithm's flat-encoding operator catalogue, under
            src/main/resources/parameterSpaces/ (same format/parser as a base
            algorithm's own `yamlParameterSpaceFile`) — mirrors Evolver's
            MetaAlgorithmRegistry.MetaAlgorithmDescriptor.operatorParameterSpaceFile.
            None when the algorithm hardcodes its operators in Java instead
            (SPEA2, SMPSO); `flat_parameters` is the fallback for those.
        tree_operator_parameter_space_file: Same as `operator_parameter_space_file`,
            for the tree encoding (Evolver's *MetaTree.yaml files, whose
            crossover/mutation are fixed to subtree/tree). None when the
            algorithm does not support tree or has no operators.
    """

    name: str
    supports_flat: bool
    supports_tree: bool
    flat_parameters: tuple[str, ...]
    tree_parameters: tuple[str, ...]
    wired_into_cli_runner: bool
    example_config_file: str | None = None
    tree_example_config_file: str | None = None
    uses_population: bool = True
    fixed_operators_note: str | None = None
    flat_only_reason: str | None = None
    operator_parameter_space_file: str | None = None
    tree_operator_parameter_space_file: str | None = None


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
        runnable_encodings=("Double", "Binary", "Permutation"),
        default_configurations={
            "Double": (("Default", "NSGAIIDoubleDefault.txt"),),
            "Binary": (("Default", "NSGAIIBinaryDefault.txt"),),
            "Permutation": (("Default", "NSGAIIPermutationDefault.txt"),),
        },
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
        runnable_encodings=("Double", "Binary", "Permutation"),
        required_extra_config_keys=("weightVectorFilesDirectory",),
        # No default configuration for Binary and Permutation: the form starts from the space.
        default_configurations={"Double": (("Default", "MOEADDoubleDefault.txt"),)},
    ),
    # org.uma.evolver.algorithm.smsemoa.{Double,Binary,Permutation}SMSEMOA
    BaseAlgorithm(
        name="SMS-EMOA",
        encodings={
            "Double": "SMSEMOADouble.yaml",
            "Binary": "SMSEMOABinary.yaml",
            "Permutation": "SMSEMOAPermutation.yaml",
        },
        runnable_today=True,
        registry_name="SMS-EMOA",
        runnable_encodings=("Double", "Binary", "Permutation"),
        default_configurations={"Double": (("Default", "SMSEMOADoubleDefault.txt"),)},
    ),
    # org.uma.evolver.algorithm.rdemoea.{Double,Permutation}RDEMOEA (no Binary variant)
    BaseAlgorithm(
        name="RDE-MOEA",
        encodings={"Double": "RDEMOEADouble.yaml", "Permutation": "RDEMOEAPermutation.yaml"},
        runnable_today=True,
        registry_name="RDEMOEA",
        runnable_encodings=("Double", "Permutation"),
    ),
    # org.uma.evolver.algorithm.agemoea.DoubleAGEMOEA (Double only)
    BaseAlgorithm(
        name="AGE-MOEA",
        encodings={"Double": "AGEMOEADouble.yaml"},
        runnable_today=True,
        registry_name="AGE-MOEA",
        runnable_encodings=("Double",),
        default_configurations={"Double": (("Default", "AGEMOEADoubleDefault.txt"),)},
    ),
    # org.uma.evolver.algorithm.rvea.DoubleRVEA (Double only)
    BaseAlgorithm(
        name="RVEA",
        encodings={"Double": "RVEADouble.yaml"},
        runnable_today=True,
        registry_name="RVEA",
        runnable_encodings=("Double",),
        required_extra_config_keys=("weightVectorFilesDirectory",),
        # The three variants of the RVEA family differ in their `replacement` parameter.
        default_configurations={
            "Double": (
                ("RVEA", "RVEADoubleDefault.txt"),
                ("RVEA*", "RVEAStarDoubleDefault.txt"),
                ("iRVEA", "IRVEADoubleDefault.txt"),
            )
        },
    ),
    # org.uma.evolver.algorithm.mopso.BaseMOPSO (Double only, particle swarm)
    BaseAlgorithm(name="MOPSO", encodings={"Double": "MOPSO.yaml"}, runnable_today=False),
    # org.uma.evolver.algorithm.nsgaiii.DoubleNSGAIII (Double only)
    BaseAlgorithm(
        name="NSGA-III",
        encodings={"Double": "NSGAIIIDouble.yaml"},
        runnable_today=True,
        registry_name="NSGA-III",
        runnable_encodings=("Double",),
        default_configurations={"Double": (("Default", "NSGAIIIDoubleDefault.txt"),)},
    ),
    # org.uma.evolver.algorithm.paes.{Double,Binary,Permutation}PAES
    BaseAlgorithm(
        name="PAES",
        encodings={
            "Double": "PAESDouble.yaml",
            "Binary": "PAESBinary.yaml",
            "Permutation": "PAESPermutation.yaml",
        },
        runnable_today=True,
        registry_name="PAES",
        runnable_encodings=("Double", "Binary", "Permutation"),
        default_configurations={
            "Double": (("Default", "PAESDoubleDefault.txt"),),
            "Binary": (("Default", "PAESBinaryDefault.txt"),),
            "Permutation": (("Default", "PAESPermutationDefault.txt"),),
        },
    ),
    # org.uma.evolver.algorithm.ssmoea.DoubleSSMOEA (Double only)
    BaseAlgorithm(
        name="SSMOEA",
        encodings={"Double": "SSMOEADouble.yaml"},
        runnable_today=True,
        registry_name="SSMOEA",
        runnable_encodings=("Double",),
        default_configurations={"Double": (("Default", "SSMOEADoubleDefault.txt"),)},
    ),
)

# Every other file under parameterSpaces/ as of this writing, explicitly triaged as NOT a base
# algorithm's own parameter space — so a drift-detection test (see tests/test_catalogue.py) can flag
# any *new* file it doesn't recognize, instead of silently ignoring it or false-alarming on these.
KNOWN_NON_ALGORITHM_PARAMETER_SPACE_FILES = frozenset(
    {
        # Meta-level parameter spaces (the search space for the *meta*-optimizer's own operators,
        # passed as metaYamlParameterSpaceFile — smaller companions to a same-named file above).
        "NSGAIIDoubleReduced.yaml",
        # An alternative base-level space of NSGA-II, the one of Nebro et al. (GECCO 2019), used by
        # Evolver's tutorial E5: a smaller companion of NSGAIIDouble.yaml, not the algorithm's own.
        "NSGAIIDoubleGECCO2019.yaml",
        "MOEADDoubleReduced.yaml",
        "RDEMOEADoubleReduced.yaml",
        "SMSEMOADoubleReduced.yaml",
        "MOPSOReduced.yaml",
        # Internal operator catalogues for the meta-optimizer itself (cli.training's
        # MetaAlgorithmRegistry and MetaSPEA2Builder), not a base-level algorithm's own parameter
        # space — not user-facing, hardcoded per registered meta-algorithm and encoding.
        "NSGAIIMetaDouble.yaml",
        "NSGAIIMetaTree.yaml",
        "AGEMOEAMetaDouble.yaml",
        "AGEMOEAMetaTree.yaml",
        "AsyncNSGAIIMetaDouble.yaml",
        "AsyncNSGAIIMetaTree.yaml",
        "SPEA2MetaDouble.yaml",
    }
)

_FLAT_NSGAII_PARAMETERS = (
    "populationSize",
    "maxEvaluations",
    "numberOfCores",
    # The rest come from MetaAlgorithmRegistry's internal NSGAIIMetaDouble.yaml catalogue
    # (algorithmResult/createInitialSolutions/variation, and offspringPopulationSize — always equal
    # to the meta population size — are fixed, not user-configurable):
    "crossover",
    "mutation",
    "selection",
)
# Tree encoding: operator flags parsed against NSGAIIMetaTree.yaml/AGEMOEAMetaTree.yaml; the
# crossover (subtree) and mutation (tree) are fixed, only their hyperparameters and the selection
# are configurable.
_TREE_NSGAII_PARAMETERS = (
    "metaPopulationSize",
    "metaMaxEvaluations",
    "numberOfCores",
    "crossoverProbability",
    "mutationProbability",
    "mutationDistributionIndex",
    "selection",
)

META_ALGORITHMS: tuple[MetaAlgorithm, ...] = (
    # Every population-based meta-optimizer generates as many offspring as its population size
    # (whole generations are evaluated in parallel) and returns its final population, never an
    # external archive; neither is configurable.
    #
    # org.uma.evolver.cli.training.MetaAlgorithmRegistry ("NSGA-II") — built on DoubleNSGAII
    # (flat, resolveFlat) and org.uma.evolver.meta.algorithm.TreeNSGAII (tree, resolveTree),
    # evaluating in parallel (MultiThreadedEvaluation).
    MetaAlgorithm(
        name="NSGA-II",
        supports_flat=True,
        supports_tree=True,
        flat_parameters=_FLAT_NSGAII_PARAMETERS,
        tree_parameters=_TREE_NSGAII_PARAMETERS,
        wired_into_cli_runner=True,
        example_config_file="MetaNSGAIIFlatConfiguration.yaml",
        tree_example_config_file="MetaNSGAIITreeConfiguration.yaml",
        operator_parameter_space_file="NSGAIIMetaDouble.yaml",
        tree_operator_parameter_space_file="NSGAIIMetaTree.yaml",
    ),
    # org.uma.evolver.cli.training.MetaAlgorithmRegistry ("AGE-MOEA") — built on DoubleAGEMOEA
    # (flat) and org.uma.evolver.meta.algorithm.TreeAGEMOEA (tree); same operator catalogue as
    # NSGA-II plus its environmental selection variant (agemoeaVariant: agemoea/agemoea2).
    MetaAlgorithm(
        name="AGE-MOEA",
        supports_flat=True,
        supports_tree=True,
        flat_parameters=(*_FLAT_NSGAII_PARAMETERS, "agemoeaVariant"),
        tree_parameters=(*_TREE_NSGAII_PARAMETERS, "agemoeaVariant"),
        wired_into_cli_runner=True,
        example_config_file="MetaAGEMOEAFlatConfiguration.yaml",
        tree_example_config_file="MetaAGEMOEATreeConfiguration.yaml",
        operator_parameter_space_file="AGEMOEAMetaDouble.yaml",
        tree_operator_parameter_space_file="AGEMOEAMetaTree.yaml",
    ),
    # org.uma.evolver.cli.training.MetaAlgorithmRegistry ("SPEA2") — built via MetaSPEA2Builder,
    # which hardcodes its own operators (SBX, polynomial mutation, KNN density estimator,
    # tournament selection); only mutationProbabilityFactor is configurable (optional).
    MetaAlgorithm(
        name="SPEA2",
        supports_flat=True,
        supports_tree=False,
        flat_parameters=(
            "populationSize",
            "maxEvaluations",
            "numberOfCores",
            "mutationProbabilityFactor",
        ),
        tree_parameters=(),
        wired_into_cli_runner=True,
        example_config_file="MetaSPEA2FlatConfiguration.yaml",
        fixed_operators_note=(
            "SPEA2 hardcodes its operators (SBX crossover, polynomial mutation, strength "
            "ranking, KNN density estimator and tournament selection): there is no catalogue to "
            "choose from. The only flag it takes is the optional `mutationProbabilityFactor` "
            "(1.0 if left out), e.g. `mutationProbabilityFactor: 1.5`."
        ),
        flat_only_reason=(
            "its operators are written for real-valued solutions, and Evolver has no tree "
            "version of it (the tree encoding has NSGA-II, AGE-MOEA, AsyncNSGA-II and Random "
            "Search)"
        ),
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
        fixed_operators_note=(
            "SMPSO hardcodes everything (perturbation mutation, velocity update, selection and "
            "archive): there is nothing to set here, and Evolver rejects any flag."
        ),
        flat_only_reason=(
            "a particle swarm moves real-valued vectors with a velocity, and a derivation tree "
            "is not one"
        ),
    ),
    # org.uma.evolver.cli.training.MetaAlgorithmRegistry ("AsyncNSGA-II") — built via
    # MetaAsyncNSGAIIBuilder (flat) and on DerivationTreeSolution (tree), hardcoding its own
    # selection/replacement; only its crossover/mutation operators are configurable, via a much
    # smaller parameter space (fixed to subtree/tree for the tree encoding).
    MetaAlgorithm(
        name="AsyncNSGA-II",
        supports_flat=True,
        supports_tree=True,
        flat_parameters=(
            "populationSize",
            "maxEvaluations",
            "numberOfCores",
            "crossover",
            "mutation",
        ),
        tree_parameters=tuple(p for p in _TREE_NSGAII_PARAMETERS if p != "selection"),
        wired_into_cli_runner=True,
        example_config_file="MetaAsyncNSGAIIFlatConfiguration.yaml",
        tree_example_config_file="MetaAsyncNSGAIITreeConfiguration.yaml",
        operator_parameter_space_file="AsyncNSGAIIMetaDouble.yaml",
        tree_operator_parameter_space_file="AsyncNSGAIIMetaTree.yaml",
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
    # MetaRandomSearchBuilder (org.uma.evolver.meta.algorithm.RandomSearch), which has no
    # population concept and exposes no operator catalogue at all (it just samples at random);
    # generic over the solution type, so wired for both flat (resolveFlatRandomSearch) and tree
    # (resolveTreeRandomSearch), where it samples random derivation trees.
    MetaAlgorithm(
        name="RandomSearch",
        supports_flat=True,
        supports_tree=True,
        flat_parameters=("maxEvaluations", "numberOfCores"),
        tree_parameters=("maxEvaluations", "numberOfCores"),
        wired_into_cli_runner=True,
        example_config_file="MetaRandomSearchFlatConfiguration.yaml",
        tree_example_config_file="MetaRandomSearchTreeConfiguration.yaml",
        uses_population=False,
    ),
)


@dataclass(slots=True, frozen=True)
class QualityIndicator:
    """A quality indicator a training run can minimize, or a validation measure.

    Attributes:
        registry_name: The name a training request uses for it, as
            org.uma.evolver.cli.IndicatorRegistry registers it.
        short_name: Its abbreviation, as jMetal names it (the column header of
            INDICATORS.csv).
        full_name: Its full name.
        measures: What it measures, in a sentence.
        maximized: Whether higher values are better. A training minimizes its meta-objectives,
            so only a minimized indicator can be one.
        in_validation: Whether Validation offers it: one that only exists so that a training can
            minimize it (the negated hypervolume) tells a validation nothing new.
    """

    registry_name: str
    short_name: str
    full_name: str
    measures: str
    maximized: bool = False
    in_validation: bool = True


# org.uma.evolver.cli.IndicatorRegistry. Evolver normalizes each front with the reference front of
# its problem and computes every indicator against that front; all of them are minimized today, but
# the analysis of a validation follows `maximized`, for an indicator where higher is better.
QUALITY_INDICATORS: tuple[QualityIndicator, ...] = (
    # org.uma.jmetal.qualityindicator.impl.Epsilon
    QualityIndicator(
        registry_name="Epsilon",
        short_name="EP",
        full_name="Additive epsilon",
        measures="The smallest amount by which the front must be shifted to weakly dominate the "
        "reference front: convergence, and coverage of its extremes.",
    ),
    # org.uma.jmetal.qualityindicator.impl.NormalizedHypervolume
    QualityIndicator(
        registry_name="NormalizedHypervolume",
        short_name="NHV",
        full_name="Normalized hypervolume",
        measures="1 − HV(front)/HV(reference front): convergence and spread together; 0 when "
        "the front covers as much of the objective space as the reference front.",
    ),
    # org.uma.jmetal.qualityindicator.impl.InvertedGenerationalDistancePlus
    QualityIndicator(
        registry_name="InvertedGenerationalDistancePlus",
        short_name="IGD+",
        full_name="Inverted generational distance plus",
        measures="The average distance from each reference point to the front, counting only "
        "the objectives the front is worse in: convergence and spread, weakly Pareto compliant.",
    ),
    # org.uma.evolver.util.HypervolumeMinus
    QualityIndicator(
        registry_name="HypervolumeMinus",
        short_name="HVMinus",
        full_name="Hypervolume, negated",
        measures="−HV(front): the hypervolume, negated so that it can be minimized like the "
        "other indicators.",
        in_validation=False,
    ),
    # org.uma.jmetal.qualityindicator.impl.Spread
    QualityIndicator(
        registry_name="Spread",
        short_name="SP",
        full_name="Spread",
        measures="Deb's diversity indicator: how evenly the solutions are distributed along the "
        "front, and how well it reaches the extremes of the reference front; not convergence. "
        "Defined only for two objectives: Evolver rejects a request that uses it on a problem "
        "with any other number.",
    ),
    # org.uma.jmetal.qualityindicator.impl.GeneralizedSpread
    QualityIndicator(
        registry_name="GeneralizedSpread",
        short_name="GSPREAD",
        full_name="Generalized spread",
        measures="The diversity of the solutions, as Spread, for any number of objectives: how "
        "evenly they are distributed and how well they reach the extremes of the reference "
        "front; not convergence.",
    ),
)


def quality_indicator(name: str) -> QualityIndicator | None:
    """A quality indicator by its registry name or its abbreviation (e.g. "Epsilon" or "EP")."""
    return next((i for i in QUALITY_INDICATORS if name in (i.registry_name, i.short_name)), None)


def is_maximized(name: str) -> bool:
    """Whether higher values of an indicator are better; False for an unknown one.

    Args:
        name: Its registry name or its abbreviation.
    """
    indicator = quality_indicator(name)
    return indicator is not None and indicator.maximized


def validation_indicators() -> tuple[QualityIndicator, ...]:
    """The quality indicators Validation offers."""
    return tuple(i for i in QUALITY_INDICATORS if i.in_validation)


def is_older_than_catalogue(version: str) -> bool:
    """Tell whether an Evolver version predates the release this catalogue mirrors.

    The catalogue mirrors the release the app runs (evolver_client.EVOLVER_VERSION): a jar older
    than it, set with EVOLVER_JAR, has outdated parameter spaces and cannot run every algorithm
    marked runnable here. Only the numeric release is compared, so a snapshot ("2.3-SNAPSHOT",
    built from develop on its way to 2.3) counts as that release.

    Args:
        version: An Evolver version, e.g. "2.1" or "2.3-SNAPSHOT".

    Returns:
        Whether it is older than EVOLVER_VERSION.
    """
    return _release_numbers(version) < _release_numbers(EVOLVER_VERSION)


def is_at_least(version: str, minimum: str) -> bool:
    """Tell whether an Evolver version is at least a given release.

    Only the numeric release is compared, so a snapshot ("2.3-SNAPSHOT") counts as its release.

    Args:
        version: An Evolver version, e.g. "2.2" or "2.3-SNAPSHOT".
        minimum: The release it must reach, e.g. "2.3".

    Returns:
        Whether `version` is `minimum` or later.
    """
    return _release_numbers(version) >= _release_numbers(minimum)


def _release_numbers(version: str) -> tuple[int, ...]:
    release = version.split("-", 1)[0]
    return tuple(int(part) for part in release.split(".") if part.isdigit())
