"""Reading the configurations a training run found, from its VAR_CONF.txt.

The file is appended to at every checkpoint of the run: a block of `# Evaluation: N` and
`# Time (min): x` lines and then one line per configuration of the front at that checkpoint, with
its meta-objectives, `|` and the configuration, as the string of `--parameter value` pairs that a
solve request takes as its `configuration`::

    # Evaluation: 2000
    # Time (min): 12.345
    EP=0.0123 NHV=0.0456 | --algorithmResult population --crossover SBX ...

The configurations a training found are those of its last checkpoint: the earlier ones are the
fronts it went through.
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


@dataclass(slots=True, frozen=True)
class Checkpoint:
    """The front of configurations at one checkpoint of a training run.

    Attributes:
        evaluation: The meta-evaluations done when it was written, or None if the block has no
            `# Evaluation` line.
        minutes: The computing time spent until then, or None if the block does not say.
        configurations: The configurations of the front, in file order.
    """

    evaluation: int | None
    minutes: float | None
    configurations: tuple[TrainedConfiguration, ...]


def parse_var_conf_checkpoints(text: str) -> list[Checkpoint]:
    """Read every checkpoint of a VAR_CONF.txt.

    Args:
        text: The file's content.

    Returns:
        The checkpoints in file order. A line that has no `|`, or whose objectives cannot be
        read, is skipped, and so is a block left with no configuration (one cut short by a read
        in the middle of a write).
    """
    blocks: list[dict] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# Evaluation:"):
            blocks.append({"evaluation": _number(stripped, int), "minutes": None, "rows": []})
        elif stripped.startswith("# Time (min):") and blocks:
            blocks[-1]["minutes"] = _number(stripped, float)
        elif stripped and not stripped.startswith("#") and "|" in stripped:
            configuration = _parse_configuration_line(stripped)
            if configuration is None:
                continue
            if not blocks:
                blocks.append({"evaluation": None, "minutes": None, "rows": []})
            blocks[-1]["rows"].append(configuration)
    return [
        Checkpoint(block["evaluation"], block["minutes"], tuple(block["rows"]))
        for block in blocks
        if block["rows"]
    ]


def parse_var_conf(text: str) -> list[TrainedConfiguration]:
    """Read the configurations a training run found: those of its last checkpoint.

    Args:
        text: The file's content.

    Returns:
        The configurations of the last checkpoint, in file order; empty if there is none.
    """
    checkpoints = parse_var_conf_checkpoints(text)
    return list(checkpoints[-1].configurations) if checkpoints else []


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


def _parse_configuration_line(line: str) -> TrainedConfiguration | None:
    objectives_text, configuration = line.split("|", 1)
    objectives = _parse_objectives(objectives_text)
    if objectives is None or not configuration.strip():
        return None
    return TrainedConfiguration(objectives, configuration.strip())


def _number(line: str, kind: type) -> int | float | None:
    try:
        return kind(line.split(":", 1)[1].strip())
    except (IndexError, ValueError):
        return None


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
