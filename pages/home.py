"""Home: what Evolver-Studio is for, a first action, the run in progress or the latest result,
and the way through its three steps (explore, solve, meta-optimize) and its tutorials.

What it shows besides the links comes from evolver_studio/home_panel.py.
"""

from pathlib import Path

import streamlit as st

from evolver_studio import home_panel
from evolver_studio.app_state import Fact, evolver_facts, java_version, render_jar_status
from evolver_studio.evolver_client import EVOLVER_VERSION, WORKING_DIRECTORY
from evolver_studio.resource_files import jar_evolver_version
from evolver_studio.runs import RunPhase
from evolver_studio.sections import SECTIONS, Page, Section
from evolver_studio.tutorial_navigation import open_tutorial
from evolver_studio.tutorials import TUTORIALS, Tutorial, TutorialLevel, tutorials_by_level

ICON = Path(__file__).resolve().parent.parent / "assets" / "logo" / "studio-icon.svg"
EVOLVER_URL = "https://github.com/jMetal/Evolver"
EVOLVER_DOCUMENTATION_URL = "https://evolver.readthedocs.io"
STEPS = ("Explore", "Solve", "Meta-optimization")
SOLVE_PAGE = "pages/solve.py"
TRAINING_PAGE = "pages/training.py"
TRAINING_ANALYSIS_PAGE = "pages/training_analysis.py"
TUTORIALS_PAGE = "pages/tutorials.py"
TOUR = "tour"
PICTURE_HEIGHT = 320
PHASE_BADGES = {
    RunPhase.STARTING: ("Starting", "blue"),
    RunPhase.RUNNING: ("Running", "blue"),
    RunPhase.FINISHED: ("Finished", "green"),
    RunPhase.FAILED: ("Failed", "red"),
    RunPhase.CANCELLED: ("Cancelled", "gray"),
    RunPhase.LOST: ("Interrupted", "orange"),
}
# The colors of the logo: navy and blue for the tile, teal for the slider, and the light green,
# cyan and white of the fronts. A stripe as the fronts (green, cyan, teal), the facts on a navy
# band, and the steps on pale blue with a teal edge. Containers with a key get the class
# st-key-<key>.
NAVY = "#0B3C6E"
BLUE = "#0E6BA8"
TEAL = "#14A3B8"
GREEN = "#7CD69B"
CYAN = "#5CC8D6"
PALE_BLUE = "#EAF4F9"
STYLE = f"""
<style>
.studio-stripe {{
  height: 6px; border-radius: 3px; margin: 0.25rem 0 0.5rem;
  background: linear-gradient(to right, {GREEN} 0 40%, {CYAN} 40% 70%, {TEAL} 70% 100%);
}}
.st-key-home_facts {{
  background: linear-gradient(135deg, {NAVY}, {BLUE}); border-radius: 0.9rem;
  padding: 1rem 1.25rem;
}}
.st-key-home_facts [data-testid="stMetric"] {{
  border-left: 4px solid {CYAN}; padding-left: 0.75rem;
}}
.st-key-home_facts [data-testid="stCaptionContainer"] {{
  color: #FFFFFF; opacity: 0.9; padding-left: 0.95rem;
}}
.st-key-home_facts [data-testid="stMetricLabel"],
.st-key-home_facts [data-testid="stMetricLabel"] *,
.st-key-home_facts [data-testid="stMetricValue"] {{ color: #FFFFFF; }}
[class*="st-key-home_step_"], [class*="st-key-home_learn_"] {{
  background: {PALE_BLUE}; border-top: 5px solid {TEAL}; border-radius: 0.9rem;
  padding: 1rem 1rem 1.25rem;
}}
[class*="st-key-home_page_"] {{ background: #FFFFFF; }}
.studio-step {{
  display: flex; align-items: center; gap: 0.6rem; font-family: "Montserrat", sans-serif;
  font-size: 1.5rem; font-weight: 700; color: {NAVY};
}}
.studio-step span {{
  display: inline-flex; align-items: center; justify-content: center; width: 2.1rem;
  height: 2.1rem; border-radius: 50%; background: {NAVY}; color: {GREEN}; font-size: 1.1rem;
}}
</style>
"""


def plural(count: int, noun: str) -> str:
    return f"{count} {noun}{'' if count == 1 else 's'}"


def start_tour() -> None:
    open_tutorial(TOUR)
    st.switch_page(TUTORIALS_PAGE)


def render_picture(summary: home_panel.Activity) -> None:
    """The run in progress, or else the front of the latest result, or else the logo."""
    if summary.running is not None:
        st.image(str(ICON), width=140)
        st.info(f"In progress: {summary.running.label} ({summary.running.kind}).")
        page = SOLVE_PAGE if summary.running.kind == home_panel.SOLVE else TRAINING_PAGE
        st.page_link(page, label="Follow it", icon=":material/monitoring:")
        return
    figure = None
    if summary.latest is not None:
        figure = home_panel.front_figure(summary.latest.front, PICTURE_HEIGHT)
    if summary.latest is None or figure is None:
        st.image(str(ICON), width=220)
        return
    st.plotly_chart(figure, config={"displayModeBar": False}, width="stretch")
    st.caption(f"Latest result: {summary.latest.label} ({summary.latest.run_id}).")
    page = SOLVE_PAGE if summary.latest.kind == home_panel.SOLVE else TRAINING_ANALYSIS_PAGE
    st.page_link(page, label="Open it", icon=":material/arrow_forward:")


def render_presentation(jar: Path | None) -> None:
    st.title("Evolver-Studio")
    st.markdown(f"#### The visual workbench for [Evolver]({EVOLVER_URL}) {EVOLVER_VERSION}")
    st.markdown(
        "Solve problems with Evolver's configurable multi-objective metaheuristics, or let it "
        "find good configurations of those algorithms for you: train, analyze, validate."
    )
    if jar is None:
        st.info(
            "**Getting started.** Evolver-Studio runs Evolver, a Java framework (Java 21 or "
            "newer is needed): download its jar with the button above. Meanwhile, the tutorials "
            f"and [Evolver's documentation]({EVOLVER_DOCUMENTATION_URL}) are a good place to "
            "start.",
            icon=":material/rocket_launch:",
        )
    if jar is not None:
        version = jar_evolver_version(jar) or EVOLVER_VERSION
        st.caption(
            f":material/check_circle: Evolver {version} · {java_version() or 'Java not found'}"
        )
    first, second, third = st.columns(3, vertical_alignment="center")
    first.page_link(
        SOLVE_PAGE,
        label="Run an algorithm",
        icon=":material/rocket_launch:",
        width="stretch",
    )
    second.page_link(
        TRAINING_PAGE, label="Start a training", icon=":material/model_training:", width="stretch"
    )
    if third.button("Take the tour", icon=":material/school:", width="stretch"):
        start_tour()


def render_recent(summary: home_panel.Activity) -> None:
    """The newest runs of either use, one line each."""
    st.markdown("**Recent runs**")
    if not summary.recent:
        st.caption("No runs yet: run an algorithm to start.")
        return
    for run in summary.recent:
        label, color = PHASE_BADGES[run.phase]
        when = run.launched.strftime("%d %b, %H:%M") if run.launched else run.run_id
        st.markdown(f"**{run.label}** :gray[{run.kind} · {when}] :{color}-badge[{label}]")


def render_facts(facts: dict[str, Fact] | None) -> None:
    written = sum(tutorial.available for tutorial in TUTORIALS)
    planned = len(TUTORIALS) - written
    shown = {
        **(facts or {}),
        "Tutorials": Fact(written, (f"{planned} more planned",) if planned else ()),
    }
    st.html('<div class="studio-stripe"></div>')
    with st.container(key="home_facts"):
        for column, (label, fact) in zip(st.columns(len(shown)), shown.items(), strict=True):
            column.metric(label, fact.count)
            column.caption(fact.detail(limit=2))


def render_page(page: Page) -> None:
    """A page of a step: a link with what it offers."""
    slug = "".join(letter if letter.isalnum() else "_" for letter in page.title.lower())
    with st.container(border=True, key=f"home_page_{slug}"):
        if page.available:
            st.page_link(page.path, label=page.title, icon=page.icon)
        else:
            st.markdown(f"{page.icon} **{page.title}** :gray-badge[Coming soon]")
        st.caption(page.description)


def step_state(name: str, summary: home_panel.Activity, facts: dict[str, Fact] | None) -> str:
    """What a step has now, in a few words."""
    if name == "Explore":
        if facts is None:
            return "Needs Evolver's jar"
        return (
            f"{plural(facts['Base algorithms'].count, 'algorithm')} · "
            f"{plural(facts['Problems'].count, 'problem')} · "
            f"{plural(facts['Quality indicators'].count, 'indicator')}"
        )
    if name == "Solve":
        return plural(summary.solve_runs, "run")
    return plural(summary.trainings, "training")


def render_step(
    number: int, section: Section, summary: home_panel.Activity, facts: dict[str, Fact] | None
) -> None:
    with st.container(key=f"home_step_{number}"):
        st.html(f'<div class="studio-step"><span>{number}</span>{section.name}</div>')
        st.caption(step_state(section.name, summary, facts))
        st.markdown(section.summary)
        for page in section.pages:
            render_page(page)


def render_tutorial(tutorial: Tutorial) -> None:
    """A tutorial of Learn: what it is for, with a button to start it if it is written."""
    with st.container(border=True, key=f"home_page_tutorial_{tutorial.slug}"):
        st.markdown(f"**{tutorial.label}**")
        st.caption(tutorial.summary)
        if not tutorial.available:
            st.markdown(":gray-badge[Coming soon]")
        elif st.button("Start", key=f"home_tutorial_{tutorial.slug}", icon=":material/school:"):
            open_tutorial(tutorial.slug)
            st.switch_page(TUTORIALS_PAGE)


def render_learn(section: Section) -> None:
    """The tutorials, by level, as the steps."""
    st.subheader(f"{section.icon} {section.name}", anchor=False)
    st.caption(section.summary)
    for number, (column, level) in enumerate(
        zip(st.columns(len(TutorialLevel), gap="large"), TutorialLevel), start=1
    ):
        with column, st.container(key=f"home_learn_{number}"):
            st.html(f'<div class="studio-step"><span>{number}</span>{level.value}</div>')
            for tutorial in tutorials_by_level(level):
                render_tutorial(tutorial)


jar = render_jar_status()
summary = home_panel.activity(WORKING_DIRECTORY)
facts = evolver_facts(str(jar)) if jar is not None else None

st.html(STYLE)
text, picture = st.columns([5, 4], vertical_alignment="top", gap="large")
with text:
    render_presentation(jar)
    render_recent(summary)
with picture:
    render_picture(summary)
render_facts(facts)
sections = {section.name: section for section in SECTIONS}
for number, (column, name) in enumerate(zip(st.columns(3, gap="large"), STEPS), start=1):
    with column:
        render_step(number, sections[name], summary, facts)
st.divider()
render_learn(sections["Learn"])
