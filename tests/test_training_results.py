"""Tests for reading the configurations a training run found."""

from pathlib import Path

from evolver_studio.training_results import (
    parse_var_conf,
    parse_var_conf_checkpoints,
    read_var_conf,
)

VAR_CONF = """# Evaluation: 100
# Time (min): 0.069
EP=5.9 NHV=0.569 | --algorithmResult population --crossover SDX --crossoverProbability 0.01
EP=14.7 NHV=0.559 | --algorithmResult population --crossover arithmetic
"""


class TestParseVarConf:
    def test_should_read_a_configuration_per_line_with_its_objectives(self):
        # Act
        configurations = parse_var_conf(VAR_CONF)

        # Assert
        assert len(configurations) == 2
        assert configurations[0].objectives == {"EP": 5.9, "NHV": 0.569}
        assert configurations[0].configuration == (
            "--algorithmResult population --crossover SDX --crossoverProbability 0.01"
        )

    def test_should_skip_the_header_and_what_it_cannot_read(self):
        # Arrange
        text = "# header\n\nno separator here\nEP=x | --a 1\nEP=1 | \nEP=2 | --b 2\n"

        # Act
        configurations = parse_var_conf(text)

        # Assert
        assert [c.configuration for c in configurations] == ["--b 2"]

    def test_should_describe_the_objectives_in_one_line(self):
        # Act
        configuration = parse_var_conf(VAR_CONF)[0]

        # Assert
        assert configuration.label == "EP=5.9 · NHV=0.569"


MULTI_CHECKPOINT = """# Evaluation: 100
# Time (min): 0.5
EP=9 NHV=1 | --crossover SBX
EP=8 NHV=2 | --crossover PCX

# Evaluation: 200
# Time (min): 1.25
EP=5 NHV=1 | --crossover blxAlpha

"""


class TestSeveralCheckpoints:
    """A training appends a block per checkpoint: the final front is the last one."""

    def test_should_read_only_the_last_checkpoint_as_the_configurations_found(self):
        # Act
        configurations = parse_var_conf(MULTI_CHECKPOINT)

        # Assert
        assert [c.configuration for c in configurations] == ["--crossover blxAlpha"]

    def test_should_read_every_checkpoint_with_its_evaluation_and_time(self):
        # Act
        checkpoints = parse_var_conf_checkpoints(MULTI_CHECKPOINT)

        # Assert
        assert [(c.evaluation, c.minutes, len(c.configurations)) for c in checkpoints] == [
            (100, 0.5, 2),
            (200, 1.25, 1),
        ]

    def test_should_ignore_a_last_block_cut_short_by_a_read_in_the_middle_of_a_write(self):
        # Arrange: Evolver is writing the checkpoint of the evaluation 300
        text = MULTI_CHECKPOINT + "# Evaluation: 300\n# Time (min): 2.0\n"

        # Act
        configurations = parse_var_conf(text)

        # Assert: the front found so far is still the one of the evaluation 200
        assert [c.configuration for c in configurations] == ["--crossover blxAlpha"]

    def test_should_read_a_file_without_checkpoint_headers_as_one(self):
        # Act
        checkpoints = parse_var_conf_checkpoints("EP=1 NHV=2 | --a 1\nEP=3 NHV=4 | --b 2\n")

        # Assert
        assert len(checkpoints) == 1 and checkpoints[0].evaluation is None
        assert len(checkpoints[0].configurations) == 2


class TestReadVarConf:
    def test_should_read_the_file(self, tmp_path: Path):
        # Arrange
        (tmp_path / "VAR_CONF.txt").write_text(VAR_CONF)

        # Act / Assert
        assert len(read_var_conf(tmp_path / "VAR_CONF.txt")) == 2

    def test_should_give_nothing_for_a_missing_file(self, tmp_path: Path):
        # Act / Assert
        assert read_var_conf(tmp_path / "VAR_CONF.txt") == []
