"""Reading Evolver's own resource files (parameter spaces, meta-optimizer configs) by name."""

from pathlib import Path

PARAMETER_SPACES_RELATIVE_DIR = Path("src/main/resources/parameterSpaces")
META_OPTIMIZER_CONFIGURATIONS_RELATIVE_DIR = Path("src/main/resources/metaOptimizerConfigurations")


def parameter_space_text(evolver_home: Path, filename: str) -> str:
    """Read a parameter space file's raw text.

    Args:
        evolver_home: Path to the Evolver checkout.
        filename: The parameter space YAML's filename, under
            src/main/resources/parameterSpaces/.

    Returns:
        That file's content.
    """
    return (evolver_home / PARAMETER_SPACES_RELATIVE_DIR / filename).read_text()


def meta_optimizer_configuration_text(evolver_home: Path, filename: str) -> str:
    """Read a meta-optimizer configuration file's raw text.

    Args:
        evolver_home: Path to the Evolver checkout.
        filename: The meta-optimizer configuration YAML's filename, under
            src/main/resources/metaOptimizerConfigurations/.

    Returns:
        That file's content.
    """
    return (evolver_home / META_OPTIMIZER_CONFIGURATIONS_RELATIVE_DIR / filename).read_text()
