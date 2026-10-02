"""Catalogue of Evolver-Studio's interactive tutorials.

Mirrors the Evolver-Studio part of Evolver's docs/proposals/tutorials.md (ids, levels, tracks and
the Evolver documentation tutorial each one pairs with). Only tutorials marked `available` have
content in this app; the rest are listed as coming soon.
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

TUTORIALS_CATALOGUE_URL = (
    "https://github.com/jMetal/Evolver/blob/develop/docs/proposals/tutorials.md"
)


class TutorialLevel(Enum):
    """How much a tutorial assumes of the reader."""

    INTRODUCTORY = "Introductory"
    INTERMEDIATE = "Intermediate"
    ADVANCED = "Advanced"


@dataclass(slots=True, frozen=True)
class Tutorial:
    """An interactive tutorial of Evolver-Studio.

    Attributes:
        tutorial_id: Its id in the tutorials catalogue (e.g. "S2").
        title: Its title.
        level: Its level.
        track: "Solving", "Meta-optimization" or "Both".
        summary: One-sentence description of what it teaches.
        pairs_with: Ids of the Evolver documentation tutorials it pairs with.
        available: Whether its content exists in this app yet.
    """

    tutorial_id: str
    title: str
    level: TutorialLevel
    track: str
    summary: str
    pairs_with: tuple[str, ...] = ()
    available: bool = False


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
        "S1",
        "A tour of Evolver-Studio",
        _INTRO,
        "Both",
        "Connecting to Evolver and the pages of the app.",
        ("E4",),
    ),
    Tutorial(
        "S2",
        "Exploring a parameter space",
        _INTRO,
        "Both",
        "Read a parameter space: its types of parameters, how they depend on each other, and "
        "which ones a configuration activates.",
        ("E1",),
        available=True,
    ),
    Tutorial(
        "S3",
        "Solving a problem with a configurable algorithm",
        _INTRO,
        "Solving",
        "Pick a problem, an algorithm and a configuration, run it and inspect the front.",
        ("E2",),
    ),
    Tutorial(
        "S4",
        "Your first guided training",
        _INTRO,
        "Meta-optimization",
        "A ready-made training run with a live front and its result files.",
        ("E3",),
    ),
    Tutorial(
        "S5",
        "Comparing configurations on a problem",
        _MID,
        "Solving",
        "Several configurations, several runs each, and their comparison.",
        ("E5", "E9"),
    ),
    Tutorial(
        "S6",
        "Using a tuned configuration",
        _MID,
        "Solving",
        "Solve a new problem with a configuration found in a training run.",
        ("E8",),
    ),
    Tutorial(
        "S7",
        "Customizing a training run",
        _MID,
        "Meta-optimization",
        "Algorithm, encoding, training set, meta-optimizer and parameter space editing.",
        ("E6", "E7", "E11"),
    ),
    Tutorial(
        "S8",
        "Analyzing training results",
        _MID,
        "Meta-optimization",
        "Checkpoints, fronts and choosing a configuration.",
        ("E8",),
    ),
    Tutorial(
        "S9",
        "Validating a configuration",
        _MID,
        "Meta-optimization",
        "Statistical comparison of a tuned configuration.",
        ("E9",),
    ),
    Tutorial(
        "S10",
        "Solving your own problem",
        _ADVANCED,
        "Solving",
        "Run an algorithm on a user-defined jMetal problem.",
        ("E2",),
    ),
    Tutorial(
        "S11",
        "Comparing meta-optimizers and encodings",
        _ADVANCED,
        "Meta-optimization",
        "Training runs with different meta-optimizers or encodings, and their comparison.",
        ("E12", "E13"),
    ),
    Tutorial(
        "S12",
        "Long-running jobs",
        _ADVANCED,
        "Both",
        "Background runs: cancelling, reconnecting and resuming the analysis.",
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
