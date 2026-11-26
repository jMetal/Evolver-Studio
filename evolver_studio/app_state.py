"""State shared across pages: the Evolver jar the app runs, and its download.

Each st.navigation page runs as its own script, so nothing at Python module
level survives between pages — only `st.session_state` does. The jar's status is
rendered once per page (cheap: a few widgets) rather than centralized, since
st.navigation has no single "shell" script that wraps every page's body.
"""

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

from evolver_studio.catalogue import is_older_than_catalogue
from evolver_studio.evolver_client import (
    EVOLVER_VERSION,
    JAR_OVERRIDE_VARIABLE,
    WORKING_DIRECTORY,
    describe,
    download_jar,
    is_jar_overridden,
    jar_path,
)
from evolver_studio.problem_catalogue import Problem, parse_problem_catalogue
from evolver_studio.resource_files import jar_evolver_version
from evolver_studio.result import Err, Ok


def render_jar_status() -> Path | None:
    """Tell what is wrong with Evolver's jar, if anything, offering to download it if missing.

    There is no sidebar: when the jar is in place and Java is found, nothing is shown.

    Returns:
        The jar's path, or None if it is not available yet.
    """
    jar = jar_path()
    if jar.is_file():
        if shutil.which("java") is None:
            st.warning(
                "Java was not found; Evolver needs Java 21 or newer. `make env` installs it in "
                "the app's Conda environment: start the app with `make run`."
            )
        return jar
    if is_jar_overridden():
        st.error(f"{JAR_OVERRIDE_VARIABLE} points at a missing file: `{jar}`")
        return None
    with st.container(border=True):
        st.warning(f"Evolver {EVOLVER_VERSION} has not been downloaded yet.")
        if st.button(f"Download Evolver {EVOLVER_VERSION} (about 130 MB)", type="primary"):
            progress = st.progress(0.0, text="Downloading from Maven Central…")
            result = download_jar(jar, lambda fraction: progress.progress(fraction))
            if isinstance(result, Err):
                st.error(result.message)
            else:
                st.rerun()
    return None


def require_evolver_jar() -> Path:
    """Show the jar's status, stopping the page if Evolver's jar is not available.

    Returns:
        The jar's path.
    """
    jar = render_jar_status()
    if jar is None:
        st.info("This page needs Evolver: download it with the button above.")
        st.stop()
    return jar


def warn_if_jar_older_than_catalogue(jar: Path) -> None:
    """Warn when the jar predates the Evolver version the catalogue mirrors.

    Its parameter spaces may then lack parameters the catalogue's Evolver has
    (e.g. RVEA's selection and replacement before 2.2), and some algorithms
    marked runnable cannot run with it. Nothing is shown when the jar records
    no version.

    Args:
        jar: Path to Evolver's jar.
    """
    version = jar_evolver_version(jar)
    if version is None or not is_older_than_catalogue(version):
        return
    st.warning(
        f"This page follows Evolver {EVOLVER_VERSION}, but the jar in use is Evolver "
        f"{version}: some parameter spaces may be incomplete, and some algorithms marked "
        f"runnable cannot run with it. Unset `{JAR_OVERRIDE_VARIABLE}` to use Evolver "
        f"{EVOLVER_VERSION}, or point it at a newer jar.",
        icon="⚠️",
    )


@st.cache_data(show_spinner=False)
def _manifest(jar_str: str) -> dict | None:
    """Run DescribeMain once per jar: it starts a JVM, too slow to repeat on every rerun.

    Args:
        jar_str: Path to Evolver's jar, as a string (cache keys must be hashable, and
            st.cache_data hashes Path objects by identity, not by value).

    Returns:
        The parsed manifest, or None if DescribeMain could not be run (e.g. Java is not
        installed).
    """
    result = describe(WORKING_DIRECTORY, Path(jar_str))
    return result.value if isinstance(result, Ok) else None


@dataclass(slots=True, frozen=True)
class Fact:
    """Something Evolver offers, counted, with a few examples.

    Attributes:
        count: How many there are.
        examples: Some of them (by name), the first ones in alphabetical order.
    """

    count: int
    examples: tuple[str, ...] = ()

    def detail(self, limit: int = 3) -> str:
        """The examples in one line, e.g. "AGE-MOEA, MOEAD, NSGA-II, …"."""
        shown = ", ".join(self.examples[:limit])
        return f"{shown}, …" if len(self.examples) > limit else shown


def evolver_facts(jar_str: str) -> dict[str, Fact] | None:
    """Count what Evolver offers, from DescribeMain's manifest.

    Args:
        jar_str: Path to Evolver's jar, as a string.

    Returns:
        The base algorithms, meta-optimizers, problems (with their families as examples) and
        quality indicators, by name, or None if DescribeMain could not be run.
    """
    manifest = _manifest(jar_str)
    if manifest is None:
        return None
    algorithms = sorted({entry["name"] for entry in manifest["baseAlgorithms"]})
    meta = sorted(entry["name"] for entry in manifest["metaAlgorithms"])
    families = sorted({entry["family"] for entry in manifest.get("problemCatalogue", [])})
    indicators = sorted(manifest["indicators"])
    return {
        "Base algorithms": Fact(len(algorithms), tuple(algorithms)),
        "Meta-optimizers": Fact(len(meta), tuple(meta)),
        "Problems": Fact(len(manifest["problems"]), tuple(families)),
        "Quality indicators": Fact(len(indicators), tuple(indicators)),
    }


@st.cache_data(show_spinner=False, ttl=60)
def java_version() -> str | None:
    """The first line `java -version` prints, or None when Java is not on the PATH.

    Cached for a minute: it starts a JVM, too slow to repeat on every rerun.
    """
    if shutil.which("java") is None:
        return None
    result = subprocess.run(["java", "-version"], capture_output=True, text=True)
    lines = (result.stderr or result.stdout).strip().splitlines()
    return lines[0] if lines else None


def registered_problem_names(jar_str: str) -> list[str] | None:
    """List the problems Evolver can resolve by name, from DescribeMain's manifest.

    Args:
        jar_str: Path to Evolver's jar, as a string.

    Returns:
        The registered problem names, sorted, or None if DescribeMain could not be run.
    """
    manifest = _manifest(jar_str)
    return sorted(manifest["problems"]) if manifest is not None else None


def registered_problems(jar_str: str) -> dict[str, Problem] | None:
    """Describe the problems Evolver can resolve by name: encoding, dimensions, arguments.

    Args:
        jar_str: Path to Evolver's jar, as a string.

    Returns:
        The problems by name, or None if DescribeMain could not be run or its manifest has no
        problem catalogue (Evolver 2.3 or older).
    """
    manifest = _manifest(jar_str)
    return parse_problem_catalogue(manifest) if manifest is not None else None
