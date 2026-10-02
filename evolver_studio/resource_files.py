"""Reading Evolver's own resource files (parameter spaces, meta-optimizer configs) by name.

They are packaged at the root of Evolver's jar, so they are read from it: the parameter
spaces shown and edited are always those of the Evolver release that runs them.
"""

import functools
import zipfile
from pathlib import Path

PARAMETER_SPACES_DIRECTORY = "parameterSpaces"
META_OPTIMIZER_CONFIGURATIONS_DIRECTORY = "metaOptimizerConfigurations"
DEFAULT_CONFIGURATIONS_DIRECTORY = "defaultConfigurations"
# Written by Maven into every jar it builds, with the version of the project.
POM_PROPERTIES_ENTRY = "META-INF/maven/org.uma.jmetal/Evolver/pom.properties"


def parameter_space_text(jar: Path, filename: str) -> str:
    """Read a parameter space file's raw text.

    Args:
        jar: Path to Evolver's fat jar.
        filename: The parameter space YAML's filename, under parameterSpaces/.

    Returns:
        That file's content.
    """
    return _jar_entry_text(jar, f"{PARAMETER_SPACES_DIRECTORY}/{filename}")


def meta_optimizer_configuration_text(jar: Path, filename: str) -> str:
    """Read a meta-optimizer configuration file's raw text.

    Args:
        jar: Path to Evolver's fat jar.
        filename: The meta-optimizer configuration YAML's filename, under
            metaOptimizerConfigurations/.

    Returns:
        That file's content.
    """
    return _jar_entry_text(jar, f"{META_OPTIMIZER_CONFIGURATIONS_DIRECTORY}/{filename}")


def default_configuration_text(jar: Path, filename: str) -> str:
    """Read a default configuration file's raw text.

    Args:
        jar: Path to Evolver's fat jar.
        filename: The configuration's filename, under defaultConfigurations/.

    Returns:
        That file's content: a configuration, "--parameter value ...".
    """
    return _jar_entry_text(jar, f"{DEFAULT_CONFIGURATIONS_DIRECTORY}/{filename}")


def jar_evolver_version(jar: Path) -> str | None:
    """Read the version of Evolver a jar was built from (e.g. "2.1" or "2.2-SNAPSHOT").

    Args:
        jar: Path to Evolver's fat jar.

    Returns:
        The version recorded in the jar's Maven pom.properties, or None if the
        jar does not record one.
    """
    try:
        text = _jar_entry_text(jar, POM_PROPERTIES_ENTRY)
    except KeyError:
        return None
    for line in text.splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip() == "version":
            return value.strip()
    return None


def _jar_entry_text(jar: Path, entry: str) -> str:
    # The modification time is part of the cache key, so a rebuilt jar is read again.
    return _cached_jar_entry_text(jar, jar.stat().st_mtime_ns, entry)


@functools.cache
def _cached_jar_entry_text(jar: Path, modification_time: int, entry: str) -> str:
    """Read one entry of a jar; cached, since opening the fat jar parses its whole index."""
    with zipfile.ZipFile(jar) as archive:
        return archive.read(entry).decode()
