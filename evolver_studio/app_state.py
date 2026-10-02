"""State shared across pages: the Evolver jar the app runs, and its download.

Each st.navigation page runs as its own script, so nothing at Python module
level survives between pages — only `st.session_state` does. This sidebar is
rendered once per page (cheap: a few widgets) rather than centralized, since
st.navigation has no single "shell" script that wraps every page's body.
"""

import shutil
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
from evolver_studio.resource_files import jar_evolver_version
from evolver_studio.result import Err, Ok


def render_sidebar() -> Path | None:
    """Show which Evolver jar the app runs, offering to download it if missing.

    Returns:
        The jar's path, or None if it is not available yet.
    """
    jar = jar_path()
    if jar.is_file():
        if is_jar_overridden():
            st.sidebar.caption(f"Evolver: `{jar}` (set by {JAR_OVERRIDE_VARIABLE})")
        else:
            st.sidebar.caption(f"Evolver {EVOLVER_VERSION}")
        if shutil.which("java") is None:
            st.sidebar.warning("Java was not found on the PATH; Evolver needs Java 21 or newer.")
        return jar
    if is_jar_overridden():
        st.sidebar.error(f"{JAR_OVERRIDE_VARIABLE} points at a missing file: `{jar}`")
        return None
    st.sidebar.warning(f"Evolver {EVOLVER_VERSION} has not been downloaded yet.")
    if st.sidebar.button(f"Download Evolver {EVOLVER_VERSION} (about 130 MB)"):
        progress = st.sidebar.progress(0.0, text="Downloading from Maven Central…")
        result = download_jar(jar, lambda fraction: progress.progress(fraction))
        if isinstance(result, Err):
            st.sidebar.error(result.message)
        else:
            st.rerun()
    return None


def require_evolver_jar() -> Path:
    """Render the sidebar, stopping the page if Evolver's jar is not available.

    Returns:
        The jar's path.
    """
    jar = render_sidebar()
    if jar is None:
        st.info("This page needs Evolver: download it from the sidebar.")
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
def registered_problem_names(jar_str: str) -> list[str] | None:
    """List the problems Evolver can resolve by name, from DescribeMain's manifest.

    Cached because DescribeMain starts a JVM, fast enough for one call but too slow to repeat on
    every rerun.

    Args:
        jar_str: Path to Evolver's jar, as a string (cache keys must be hashable, and
            st.cache_data hashes Path objects by identity, not by value).

    Returns:
        The registered problem names, sorted, or None if DescribeMain could not be run (e.g. Java
        is not installed).
    """
    result = describe(WORKING_DIRECTORY, Path(jar_str))
    return sorted(result.value["problems"]) if isinstance(result, Ok) else None
