"""Tutorial "A tour of Evolver-Studio": its steps.

The interactive counterpart of Evolver's tutorial E4 (docs/tutorials/automating_with_the_cli.rst):
what Evolver-Studio does for you, how it connects to Evolver, and what each page is for. It runs
nothing long: it checks the connection (Java and the jar) and asks Evolver what it can resolve,
and what the text says about Evolver comes from that answer, not from a list written down.
"""

import shutil
import subprocess
from pathlib import Path

import streamlit as st

from evolver_studio.app_state import _manifest
from evolver_studio.evolver_client import EVOLVER_VERSION, is_jar_overridden
from evolver_studio.resource_files import jar_evolver_version
from evolver_studio.runs import RUNS_DIRECTORY_NAME, SOLVE_RUNS_DIRECTORY_NAME
from evolver_studio.sections import SECTIONS
from evolver_studio.tutorial_navigation import open_tutorial
from evolver_studio.tutorials import TUTORIALS, TutorialStep
from evolver_studio.validation_runner import VALIDATION_RUNS_DIRECTORY_NAME

E4_TUTORIAL_URL = (
    "https://github.com/jMetal/Evolver/blob/develop/docs/tutorials/automating_with_the_cli.rst"
)
KEY_PREFIX = "tutorial_tour"


def _java_version() -> str | None:
    """The first line `java -version` prints, or None when Java is not on the PATH."""
    if shutil.which("java") is None:
        return None
    result = subprocess.run(["java", "-version"], capture_output=True, text=True)
    lines = (result.stderr or result.stdout).strip().splitlines()
    return lines[0] if lines else None


def _render_introduction(jar: Path) -> None:
    st.markdown(
        "Evolver-Studio is a graphical front end for **Evolver**, a Java framework that "
        "*meta-optimizes* multi-objective metaheuristics: instead of setting the parameters of an "
        "algorithm such as NSGA-II by hand, Evolver searches for good ones. Evolver has no "
        "interface of its own; it is driven by configuration files and a command line, and "
        "Evolver-Studio writes those files for you, runs Evolver, follows it, and reads what it "
        "leaves.\n\n"
        "It has **two uses**, and each page belongs to one or both:\n\n"
        "- **Solving problems.** Pick a problem and one of Evolver's configurable algorithms, "
        "adjust its configuration, run it and inspect the front it finds, as with jMetal's "
        "runners.\n"
        "- **Meta-optimization.** Let Evolver find a good configuration of an algorithm for a "
        "family of problems (*training*), study what it found (*analysis*), and check that it "
        "is really better than the defaults on problems it never saw (*validation*).\n\n"
        "This tour shows how the app connects to Evolver and what each page is for. Evolver's "
        f"tutorial [E4]({E4_TUTORIAL_URL}) shows the same machinery from the command line."
    )


def _render_connection(jar: Path) -> None:
    java = _java_version()
    version = jar_evolver_version(jar)
    st.markdown(
        "Evolver-Studio needs two things on your computer, and the **sidebar** reports on "
        "both:\n\n"
        "1. **Java 21 or newer**, which runs Evolver.\n"
        "2. **Evolver's jar**, one file with the framework and its configurations. The app "
        f"follows the release **Evolver {EVOLVER_VERSION}**; if the jar is missing, the sidebar "
        "offers to download it from Maven Central into the app's `lib/` folder. Nothing else of "
        "Evolver is needed: the parameter spaces and the meta-optimizers' configurations are "
        "read from the jar.\n\nYours:"
    )
    columns = st.columns(2)
    columns[0].metric("Java", java or "not found", help="The first line of `java -version`.")
    columns[1].metric("Evolver jar", version or "unknown version", help=f"`{jar.name}`")
    if java is None:
        st.warning("Java was not found on the PATH: install Java 21 or newer to run Evolver.")
    if is_jar_overridden():
        st.info(
            "The jar in use is the one named by the `EVOLVER_JAR` environment variable, "
            "such as a build of Evolver's development branch."
        )
    st.markdown(
        "The app never links to Evolver's code. It starts Evolver as a **separate process** and "
        "talks to it through files: it writes a *request* (what to run), Evolver keeps a "
        "*status* file up to date (running, finished, failed, how far it got), and when it ends "
        "it points to its *results*. That is why a long run survives closing the browser, and "
        "why you can read the same files yourself."
    )


def _render_what_evolver_offers(jar: Path) -> None:
    manifest = _manifest(str(jar))
    st.markdown(
        "Evolver can describe itself: the app asks the jar what it can resolve, and every page "
        "offers exactly that."
    )
    if manifest is None:
        st.warning("Evolver could not be asked: check that Java is installed.")
        return
    algorithms = sorted({entry["name"] for entry in manifest["baseAlgorithms"]})
    columns = st.columns(4)
    columns[0].metric("Base algorithms", len(algorithms))
    columns[1].metric("Meta-optimizers", len(manifest["metaAlgorithms"]))
    columns[2].metric("Problems", len(manifest["problems"]))
    columns[3].metric("Quality indicators", len(manifest["indicators"]))
    st.markdown(
        f"- **Base algorithms** are the ones that get configured: {', '.join(algorithms)}. Each "
        "has a *parameter space*: the parameters that define it, their types and ranges, and "
        "which depend on which.\n"
        "- **Meta-optimizers** are the ones that search that space (Evolver itself uses "
        "metaheuristics for this).\n"
        "- **Problems** are the benchmark and real-world problems to solve or to train on.\n"
        "- **Quality indicators** measure how good a front is; a training minimizes them."
    )
    st.page_link("pages/explore_base_algorithms.py", label="Explore the base algorithms", icon="🧬")


def _render_pages(jar: Path) -> None:
    st.markdown(
        "The menu groups the pages by purpose. A typical meta-optimization goes through them "
        "from top to bottom: explore, train, analyze, validate."
    )
    for section in SECTIONS:
        with st.container(border=True):
            st.markdown(f"**{section.name}**")
            for page in section.pages:
                st.page_link(page.path, label=f"{page.title} — {page.description}", icon=page.icon)
    st.markdown(
        "A path through them for **solving**: *Explore › Problems* to choose a problem, then "
        "*Run algorithm*. For **meta-optimization**: *Training* to launch a run, *Analysis* to "
        "read what it found, *Validation* to put a configuration to the test. Each page can "
        "hand its result to the next: a configuration goes from Training or Analysis to "
        "Validation, or to Run algorithm, with a button."
    )


def _render_where_things_live(jar: Path) -> None:
    st.markdown(
        "Everything a run produces stays in folders next to the app, so that it survives "
        "restarts and can be read with any tool:\n\n"
        f"- `{RUNS_DIRECTORY_NAME}/`: one folder per training, with the request it ran, its "
        "status and its log;\n"
        "- `results/`: what Evolver wrote for each training (`INDICATORS.csv`, `VAR_CONF.txt`, "
        "`METADATA.txt`…), which *Analysis* reads;\n"
        f"- `{SOLVE_RUNS_DIRECTORY_NAME}/`: the runs of *Run algorithm*;\n"
        f"- `{VALIDATION_RUNS_DIRECTORY_NAME}/`: the studies of *Validation*.\n\n"
        "A run keeps going when you leave its page or close the tab, and you find it again "
        "when you come back. To stop one, use its page's *Cancel* button."
    )


def _render_next_steps(jar: Path) -> None:
    st.markdown(
        "**Where to go next**, in the order the tutorials build on each other:\n\n"
        "- *Exploring a parameter space*: read what an algorithm lets you configure.\n"
        "- *Solving a problem with a configurable algorithm*: your first run.\n"
        "- *Analyzing training results*: read what a training leaves, on a real one that "
        "ships with the app.\n"
        "- *Validating a configuration*: compare a tuned configuration with the defaults.\n\n"
        "The tutorials still to be written are listed in the catalogue without a number."
    )
    written = [t.label for t in TUTORIALS if t.available]
    st.caption(f"Written so far: {'; '.join(written)}.")
    st.button(
        "All the tutorials",
        icon="🎓",
        on_click=open_tutorial,
        args=(None,),
        key=f"{KEY_PREFIX}_all_tutorials",
    )


STEPS: tuple[TutorialStep, ...] = (
    TutorialStep("What is Evolver-Studio?", _render_introduction),
    TutorialStep("Step 1: connecting to Evolver", _render_connection),
    TutorialStep("Step 2: what Evolver offers", _render_what_evolver_offers),
    TutorialStep("Step 3: the pages", _render_pages),
    TutorialStep("Step 4: where things are kept", _render_where_things_live),
    TutorialStep("Where to go next", _render_next_steps),
)
