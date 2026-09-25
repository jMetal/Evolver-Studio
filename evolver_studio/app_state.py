"""State shared across pages: the Evolver jar the app runs, and its download.

Each st.navigation page runs as its own script, so nothing at Python module
level survives between pages — only `st.session_state` does. This sidebar is
rendered once per page (cheap: a few widgets) rather than centralized, since
st.navigation has no single "shell" script that wraps every page's body.
"""

import shutil
from pathlib import Path

import streamlit as st

from evolver_studio.evolver_client import (
    EVOLVER_VERSION,
    JAR_OVERRIDE_VARIABLE,
    download_jar,
    is_jar_overridden,
    jar_path,
)
from evolver_studio.result import Err


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
