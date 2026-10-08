"""Tests for the Analysis page, on synthetic training runs."""

from pathlib import Path

import pytest
import yaml
from streamlit.testing.v1 import AppTest

from evolver_studio import evolver_client
from evolver_studio.evolver_client import jar_path

APP_SCRIPT = Path(__file__).resolve().parent.parent / "app.py"
RUN_ID = "20261007-120000"
INDICATORS = "Evaluation,SolutionId,EP,NHV\n" + "".join(
    f"{evaluation},{i},{0.5 / evaluation * 100 + i * 0.01},{0.9 - i * 0.01}\n"
    for evaluation in (50, 100)
    for i in range(3)
)
VAR_CONF = (
    "# Evaluation: 100\n"
    "# Time (min): 1.0\n"
    "EP=0.1 NHV=0.2 | --crossover SBX --crossoverProbability 0.9\n"
    "EP=0.2 NHV=0.1 | --crossover SBX --crossoverProbability 0.8\n"
    "EP=0.3 NHV=0.05 | --crossover BLX_ALPHA --crossoverProbability 0.95\n"
)


def _write_training(working_directory: Path) -> None:
    run_dir = working_directory / "cli-runner-runs" / RUN_ID
    run_dir.mkdir(parents=True)
    output = working_directory / "results" / RUN_ID
    output.mkdir(parents=True)
    (run_dir / "status.yaml").write_text(
        yaml.safe_dump(
            {"status": "FINISHED", "evaluationsDone": 100, "maxEvaluations": 100, "updatedAt": "x"}
        )
    )
    (run_dir / "request.yaml").write_text(yaml.safe_dump({"outputDirectory": f"results/{RUN_ID}"}))
    (run_dir / "base_level.yaml").write_text(
        yaml.safe_dump(
            {
                "algorithmName": "NSGA-II",
                "encoding": "Double",
                "trainingProblemNames": ["ZDT4"],
                "indicatorNames": ["Epsilon", "NormalizedHypervolume"],
            }
        )
    )
    (output / "VAR_CONF.txt").write_text(VAR_CONF)
    (output / "INDICATORS.csv").write_text(INDICATORS)


@pytest.fixture
def working_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    if not jar_path().is_file():
        pytest.skip(f"Evolver jar not found at {jar_path()}")
    monkeypatch.setattr(evolver_client, "WORKING_DIRECTORY", tmp_path)
    return tmp_path


def _open(timeout: int = 60) -> AppTest:
    return AppTest.from_file(str(APP_SCRIPT), default_timeout=timeout).switch_page(
        "pages/analysis.py"
    )


class TestAnalysisPage:
    def test_should_say_there_is_nothing_to_analyze_without_trainings(
        self, working_directory: Path
    ):
        app = _open().run()

        assert not app.exception
        assert any("no finished training run" in info.value for info in app.info)

    def test_should_show_a_finished_training_in_tabs(self, working_directory: Path):
        _write_training(working_directory)

        app = _open().run()

        assert not app.exception
        assert [tab.label for tab in app.tabs] == [
            "Convergence",
            "Front",
            "Configurations",
            "Parameters",
        ]
        assert app.selectbox(key="analysis_training").value.run_id == RUN_ID

    def test_should_say_what_the_front_does_with_each_parameter(self, working_directory: Path):
        _write_training(working_directory)

        app = _open().run()

        tables = [d.value for d in app.dataframe if "Agreement" in getattr(d.value, "columns", [])]
        assert not app.exception
        assert tables
        assert "crossover" in set(tables[0]["Parameter"])

    def test_should_send_a_configuration_to_validation(self, working_directory: Path):
        _write_training(working_directory)
        app = _open().run()

        next(b for b in app.button if b.label == "Validate it").click()
        app.run()

        assert not app.exception
