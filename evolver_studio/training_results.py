"""Reading the configurations a training run found, from its VAR_CONF.txt.

The file has a header of `# ...` lines and then one line per configuration of the front the
training ended with: its meta-objectives, `|` and the configuration, as the string of `--parameter
value` pairs that a solve request takes as its `configuration`::

    # Evaluation: 2000
    EP=0.0123 NHV=0.0456 | --algorithmResult population --crossover SBX ...
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True, frozen=True)
class TrainedConfiguration:
    """A configuration a training run found.

    Attributes:
        objectives: Its value of each meta-objective (the training's indicators), by name.
        configuration: The configuration, "--parameter value ...".
    """

    objectives: dict[str, float]
    configuration: str

    @property
    def label(self) -> str:
        """The objectives in one line, e.g. "EP=0.0123 · NHV=0.0456"."""
        return " · ".join(f"{name}={value:.4g}" for name, value in self.objectives.items())


def parse_var_conf(text: str) -> list[TrainedConfiguration]:
    """Read the configurations of a VAR_CONF.txt.

    Args:
        text: The file's content.

    Returns:
        The configurations in file order. A line that has no `|`, or whose objectives cannot be
        read, is skipped.
    """
    configurations = []
    for line in text.splitlines():
        if not line.strip() or line.startswith("#") or "|" not in line:
            continue
        objectives_text, configuration = line.split("|", 1)
        objectives = _parse_objectives(objectives_text)
        if objectives is None or not configuration.strip():
            continue
        configurations.append(TrainedConfiguration(objectives, configuration.strip()))
    return configurations


def read_var_conf(var_conf_file: Path) -> list[TrainedConfiguration]:
    """Read a training run's VAR_CONF.txt.

    Args:
        var_conf_file: The file.

    Returns:
        The configurations, empty if the file is missing or unreadable.
    """
    try:
        return parse_var_conf(var_conf_file.read_text())
    except OSError:
        return []


def _parse_objectives(text: str) -> dict[str, float] | None:
    objectives = {}
    for pair in text.split():
        name, separator, value = pair.partition("=")
        if not separator:
            return None
        try:
            objectives[name] = float(value)
        except ValueError:
            return None
    return objectives or None
