"""Tests for reading the configurations a training run found."""

from pathlib import Path

from evolver_studio.training_results import parse_var_conf, read_var_conf

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


class TestReadVarConf:
    def test_should_read_the_file(self, tmp_path: Path):
        # Arrange
        (tmp_path / "VAR_CONF.txt").write_text(VAR_CONF)

        # Act / Assert
        assert len(read_var_conf(tmp_path / "VAR_CONF.txt")) == 2

    def test_should_give_nothing_for_a_missing_file(self, tmp_path: Path):
        # Act / Assert
        assert read_var_conf(tmp_path / "VAR_CONF.txt") == []
