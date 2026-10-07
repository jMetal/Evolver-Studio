"""Catalogue of Evolver-Studio's interactive tutorials.

Numbered as Evolver numbers its own (docs/tutorials/index.rst): only the tutorials that are written
have a number, S1, S2..., consecutive in the order of the catalogue (by level), and the planned
ones are listed without one. A tutorial is identified by its topic (`slug`), which does not change
when the tutorials are renumbered. Each one pairs with the tutorials of Evolver's documentation
that cover the same ground with Java code (E1, E2..., Evolver's own numbers).
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

EVOLVER_TUTORIALS_URL = "https://github.com/jMetal/Evolver/blob/develop/docs/tutorials/index.rst"


class TutorialLevel(Enum):
    """How much a tutorial assumes of the reader."""

    INTRODUCTORY = "Introductory"
    INTERMEDIATE = "Intermediate"
    ADVANCED = "Advanced"


@dataclass(slots=True, frozen=True)
class Tutorial:
    """An interactive tutorial of Evolver-Studio.

    Attributes:
        slug: Its topic, which identifies it (e.g. "solving").
        number: Its number (e.g. "S2") when it is written, None while it is only planned.
        title: Its title.
        level: Its level.
        track: "Solving", "Meta-optimization" or "Both".
        summary: One-sentence description of what it teaches.
        pairs_with: Numbers of the Evolver documentation tutorials it pairs with.
    """

    slug: str
    number: str | None
    title: str
    level: TutorialLevel
    track: str
    summary: str
    pairs_with: tuple[str, ...] = ()

    @property
    def available(self) -> bool:
        """Whether its content exists in this app: only a written tutorial has a number."""
        return self.number is not None

    @property
    def label(self) -> str:
        """Its number and title (e.g. "S2. Solving a problem..."), or the title alone."""
        return f"{self.number}. {self.title}" if self.number else self.title


@dataclass(slots=True, frozen=True)
class TutorialStep:
    """One step of a tutorial.

    Attributes:
        title: The step's title.
        render: Renders the step's content, given Evolver's jar.
    """

    title: str
    render: Callable[[Path], None]


_INTRO = TutorialLevel.INTRODUCTORY
_MID = TutorialLevel.INTERMEDIATE
_ADVANCED = TutorialLevel.ADVANCED

TUTORIALS: tuple[Tutorial, ...] = (
    Tutorial(
        "tour",
        None,
        "A tour of Evolver-Studio",
        _INTRO,
        "Both",
        "Connecting to Evolver and the pages of the app.",
        ("E4",),
    ),
    Tutorial(
        "parameter_spaces",
        "S1",
        "Exploring a parameter space",
        _INTRO,
        "Both",
        "Read a parameter space: its types of parameters, how they depend on each other, and "
        "which ones a configuration activates.",
        ("E1",),
    ),
    Tutorial(
        "solving",
        "S2",
        "Solving a problem with a configurable algorithm",
        _INTRO,
        "Solving",
        "Pick a problem, an algorithm and a configuration, run it and inspect the front.",
        ("E2",),
    ),
    Tutorial(
        "first_training",
        None,
        "Your first guided training",
        _INTRO,
        "Meta-optimization",
        "A ready-made training run with a live front and its result files.",
        ("E3",),
    ),
    Tutorial(
        "comparing_configurations",
        None,
        "Comparing configurations on a problem",
        _MID,
        "Solving",
        "Several configurations, several runs each, and their comparison.",
        ("E8", "E12"),
    ),
    Tutorial(
        "using_a_tuned_configuration",
        None,
        "Using a tuned configuration",
        _MID,
        "Solving",
        "Solve a new problem with a configuration found in a training run.",
        ("E7",),
    ),
    Tutorial(
        "customizing_a_training",
        None,
        "Customizing a training run",
        _MID,
        "Meta-optimization",
        "Algorithm, encoding, training set, meta-optimizer and parameter space editing.",
        ("E5", "E6", "E10"),
    ),
    Tutorial(
        "analyzing_training_results",
        None,
        "Analyzing training results",
        _MID,
        "Meta-optimization",
        "Checkpoints, fronts and choosing a configuration.",
        ("E7",),
    ),
    Tutorial(
        "validating_a_configuration",
        "S3",
        "Validating a configuration",
        _MID,
        "Meta-optimization",
        "Run a tuned configuration many times next to other algorithms, and compare them with "
        "medians, a Wilcoxon test and an effect size.",
        ("E8",),
    ),
    Tutorial(
        "your_own_problem",
        None,
        "Solving your own problem",
        _ADVANCED,
        "Solving",
        "Run an algorithm on a user-defined jMetal problem.",
        ("E2", "E17"),
    ),
    Tutorial(
        "meta_optimizers_and_encodings",
        None,
        "Comparing meta-optimizers and encodings",
        _ADVANCED,
        "Meta-optimization",
        "Training runs with different meta-optimizers or encodings, and their comparison.",
        ("E13", "E14"),
    ),
    Tutorial(
        "long_running_jobs",
        None,
        "Long-running jobs",
        _ADVANCED,
        "Both",
        "Background runs: cancelling, reconnecting and resuming the analysis.",
        ("E16",),
    ),
)


def tutorials_by_level(level: TutorialLevel) -> list[Tutorial]:
    """Return the tutorials of a level, in catalogue order.

    Args:
        level: The level to select.

    Returns:
        The tutorials of that level.
    """
    return [tutorial for tutorial in TUTORIALS if tutorial.level is level]
