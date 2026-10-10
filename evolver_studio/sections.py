"""The app's pages, grouped by purpose: the single source of the menu and the home page.

`app.py` builds `st.navigation` from `SECTIONS`, and the home page (`pages/home.py`) shows a card
for each page, so the two cannot drift apart.
"""

from dataclasses import dataclass

HOME_PAGE = "pages/home.py"


@dataclass(slots=True, frozen=True)
class Page:
    """A page of the app.

    Attributes:
        path: Its script, relative to the app's root (e.g. "pages/training.py").
        title: Its title in the menu and on its card.
        icon: Its icon in the menu and on its card.
        description: What it does, for its card on the home page.
        available: Whether it is implemented, or only a placeholder.
    """

    path: str
    title: str
    icon: str
    description: str
    available: bool


@dataclass(slots=True, frozen=True)
class Section:
    """A group of pages in the menu.

    Attributes:
        name: Its heading in the menu, and the label of its pages' cards on the home page.
        pages: Its pages, in menu order.
    """

    name: str
    pages: tuple[Page, ...]


SECTIONS = (
    Section(
        "Explore",
        (
            Page(
                "pages/explore_base_algorithms.py",
                "Base algorithms",
                "🧬",
                "Browse the parameter space of every configurable algorithm, for each encoding, "
                "and see which ones can be run from the app.",
                available=True,
            ),
            Page(
                "pages/explore_meta_optimizers.py",
                "Meta-optimizers",
                "🎛️",
                "See the encodings each meta-optimizer supports and the operators it can be "
                "configured with.",
                available=True,
            ),
            Page(
                "pages/explore_quality_indicators.py",
                "Quality indicators",
                "📏",
                "The quality indicators a training run can minimize, and what each one measures.",
                available=True,
            ),
            Page(
                "pages/explore_problems.py",
                "Problems",
                "🧩",
                "The benchmark and real-world problems available for training and solving, with "
                "their encoding, objectives, variables, arguments and reference fronts.",
                available=True,
            ),
        ),
    ),
    Section(
        "Solve",
        (
            Page(
                "pages/solve.py",
                "Run algorithm",
                "▶️",
                "Choose a problem and an algorithm, start from its default configuration and "
                "adjust it within the parameter space, run it, and inspect the fronts and the "
                "quality indicators.",
                available=True,
            ),
        ),
    ),
    Section(
        "Meta-optimization",
        (
            Page(
                "pages/training.py",
                "Training",
                "🏋️",
                "Configure, launch and monitor a training run, with a live view of the front of "
                "configurations found.",
                available=True,
            ),
            Page(
                "pages/training_analysis.py",
                "Training analysis",
                "📊",
                "Study a finished training run: how it converged, how its population evolved, the "
                "configurations of its final front and what they have in common, and take one to "
                "Validation or to Run algorithm.",
                available=True,
            ),
            Page(
                "pages/validation.py",
                "Validation",
                "✅",
                "Compare a tuned configuration with the default configurations of other algorithms "
                "on a set of problems: many independent runs, medians, Wilcoxon tests and effect "
                "sizes.",
                available=True,
            ),
        ),
    ),
    Section(
        "Learn",
        (
            Page(
                "pages/tutorials.py",
                "Tutorials",
                "🎓",
                "Interactive, step-by-step tutorials that pair with Evolver's documentation.",
                available=True,
            ),
        ),
    ),
)
