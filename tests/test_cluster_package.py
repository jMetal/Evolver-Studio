"""Tests for the packages that run a training on a cluster, and for importing their results."""

import os
import shutil
import stat
import subprocess
import zipfile
from pathlib import Path

import pytest
import yaml

from evolver_studio.cluster_package import (
    MANIFEST_FILE_NAME,
    SlurmSettings,
    TrainingPackage,
    import_cluster_results,
    slurm_settings_errors,
    write_cluster_package,
)
from evolver_studio.request import BaseLevelConfig, MetaSearchConfig
from evolver_studio.result import Err, Ok
from evolver_studio.training_runs import list_finished_trainings

RUN_ID = "20261126-120000"
NAME = f"tutorial.{RUN_ID}"


def _working_directory(root: Path) -> Path:
    """A working directory with the files a MOEA/D training on ZDT1 reads."""
    working_directory = root / "studio"
    (working_directory / "resources" / "referenceFronts").mkdir(parents=True)
    (working_directory / "resources" / "referenceFronts" / "ZDT1.csv").write_text("0,1\n1,0\n")
    (working_directory / "resources" / "weightVectors").mkdir()
    (working_directory / "resources" / "weightVectors" / "W2D_100.dat").write_text("0 1\n")
    return working_directory


def _package(front: str = "resources/referenceFronts/ZDT1.csv") -> TrainingPackage:
    base_level = BaseLevelConfig(
        algorithm_name="MOEAD",
        encoding="Double",
        population_size=100,
        number_of_independent_runs=1,
        yaml_parameter_space_file="base_parameter_space.yaml",
        extra_config={"weightVectorFilesDirectory": "resources/weightVectors"},
        training_problem_names=["ZDT1"],
        training_reference_front_file_names=[front],
        training_evaluations=[2000],
        indicator_names=["Epsilon", "NormalizedHypervolume"],
    )
    meta_search = MetaSearchConfig(
        algorithm="NSGA-II",
        meta_max_evaluations=40,
        meta_population_size=10,
        number_of_cores=12,
        operator_flags={"crossover": "SBX"},
        encoding="flat",
        meta_max_computing_time_minutes=None,
    )
    return TrainingPackage(
        experiment="tutorial",
        run_id=RUN_ID,
        results_folder="results/MOEAD",
        base_level=base_level,
        meta_search=meta_search,
        parameter_space_text="algorithmResult: {}\n",
        write_frequency=20,
        write_population=True,
    )


def _write(tmp_path: Path, package: TrainingPackage | None = None, **settings) -> Path:
    jar = tmp_path / "Evolver-2.7-jar-with-dependencies.jar"
    jar.write_bytes(b"jar")
    return write_cluster_package(
        package or _package(),
        SlurmSettings(**settings),
        jar,
        _working_directory(tmp_path) if not (tmp_path / "studio").exists() else tmp_path / "studio",
        tmp_path / "exports",
    )


def _unzip(zip_file: Path, destination: Path) -> Path:
    """Unzip a package as unzip does, keeping the permissions of its files."""
    with zipfile.ZipFile(zip_file) as archive:
        for info in archive.infolist():
            path = Path(archive.extract(info, destination))
            mode = info.external_attr >> 16
            if mode:
                path.chmod(stat.S_IMODE(mode))
    return destination / NAME


class TestSlurmSettings:
    def test_should_accept_the_defaults(self):
        assert slurm_settings_errors(SlurmSettings()) == []

    @pytest.mark.parametrize("time_limit", ["100:00:00", "4-00:00:00", "90", "2-12"])
    def test_should_accept_the_formats_of_slurm_for_the_time(self, time_limit: str):
        assert slurm_settings_errors(SlurmSettings(time_limit=time_limit)) == []

    def test_should_reject_a_time_slurm_does_not_read(self):
        errors = slurm_settings_errors(SlurmSettings(time_limit="two days"))

        assert len(errors) == 1 and "time limit" in errors[0]

    def test_should_reject_no_replications_a_wrong_memory_and_an_option_without_hyphen(self):
        settings = SlurmSettings(replications=0, memory="a lot", extra_options="partition=long")

        assert len(slurm_settings_errors(settings)) == 3


class TestWriteClusterPackage:
    def test_should_hold_the_jar_the_training_its_files_and_the_scripts(self, tmp_path: Path):
        # Act
        zip_file = _write(tmp_path)

        # Assert
        assert zip_file == tmp_path / "exports" / f"{NAME}.zip"
        names = set(zipfile.ZipFile(zip_file).namelist())
        assert {
            f"{NAME}/Evolver-2.7-jar-with-dependencies.jar",
            f"{NAME}/experiment/request.yaml",
            f"{NAME}/experiment/base_level.yaml",
            f"{NAME}/experiment/meta_search.yaml",
            f"{NAME}/experiment/base_parameter_space.yaml",
            f"{NAME}/resources/referenceFronts/ZDT1.csv",
            f"{NAME}/resources/weightVectors/W2D_100.dat",
            f"{NAME}/train.sbatch",
            f"{NAME}/submit.sh",
            f"{NAME}/README.txt",
            f"{NAME}/{MANIFEST_FILE_NAME}",
        } == names

    def test_should_write_the_training_with_paths_relative_to_the_package(self, tmp_path: Path):
        # Act
        package = _unzip(_write(tmp_path), tmp_path / "cluster")

        # Assert
        request = yaml.safe_load((package / "experiment" / "request.yaml").read_text())
        base_level = yaml.safe_load((package / "experiment" / "base_level.yaml").read_text())
        assert request["baseLevel"] == "experiment/base_level.yaml"
        assert request["metaSearch"] == "experiment/meta_search.yaml"
        assert request["writePopulation"] is True
        assert base_level["yamlParameterSpaceFile"] == "experiment/base_parameter_space.yaml"
        assert base_level["trainingReferenceFrontFileNames"] == [
            "resources/referenceFronts/ZDT1.csv"
        ]

    def test_should_ask_slurm_for_the_cores_of_the_meta_optimizer(self, tmp_path: Path):
        # Act
        package = _unzip(
            _write(tmp_path, replications=5, extra_options="--partition=long"), tmp_path / "c"
        )

        # Assert
        script = (package / "train.sbatch").read_text()
        assert "#SBATCH --cpus-per-task=12\n" in script
        assert "#SBATCH --partition=long\n" in script
        assert "REPLICATIONS=5\n" in (package / "submit.sh").read_text()
        assert os.access(package / "train.sbatch", os.X_OK)
        assert os.access(package / "submit.sh", os.X_OK)

    def test_should_copy_a_front_given_by_absolute_path_and_remember_it(self, tmp_path: Path):
        # Arrange
        front = tmp_path / "elsewhere" / "Mine.csv"
        front.parent.mkdir()
        front.write_text("0,1\n")

        # Act
        package = _unzip(_write(tmp_path, _package(str(front))), tmp_path / "cluster")

        # Assert
        base_level = yaml.safe_load((package / "experiment" / "base_level.yaml").read_text())
        inside = base_level["trainingReferenceFrontFileNames"][0]
        assert inside == "resources/external/1-Mine.csv"
        assert (package / inside).read_text() == "0,1\n"
        manifest = yaml.safe_load((package / MANIFEST_FILE_NAME).read_text())
        assert manifest["pathMap"] == {inside: str(front)}

    def test_should_refuse_a_training_that_reads_a_missing_file(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="Missing.csv"):
            _write(tmp_path, _package("resources/referenceFronts/Missing.csv"))


def _fake_tools(directory: Path) -> Path:
    """A java that writes a finished training to its --output-dir, and an sbatch that echoes."""
    directory.mkdir()
    (directory / "java").write_text(
        "#!/bin/sh\n"
        'case "$1" in -XshowSettings*) echo "    java.specification.version = 21" >&2; exit 0;; '
        "esac\n"
        'while [ "$1" != "--output-dir" ]; do shift; done\n'
        'mkdir -p "$2" && echo "outputDirectory: $2" > "$2/results.yaml"\n'
        '[ -z "$FAIL" ] || exit 3\n'
    )
    (directory / "sbatch").write_text('#!/bin/sh\necho "sbatch $*"\n')
    for tool in ("java", "sbatch"):
        (directory / tool).chmod(0o755)
    return directory


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is not installed")
class TestScripts:
    @pytest.fixture
    def package(self, tmp_path: Path) -> Path:
        return _unzip(_write(tmp_path, replications=3), tmp_path / "cluster")

    def _run(self, package: Path, tmp_path: Path, *command: str, **variables: str):
        tools = tmp_path / "tools"
        if not tools.exists():
            _fake_tools(tools)
        environment = {
            "PATH": f"{tools}:{os.environ['PATH']}",
            "HOME": str(tmp_path),
            "JAVA_SETUP": "",
            **variables,
        }
        return subprocess.run(
            command, cwd=package, env=environment, capture_output=True, text=True, check=False
        )

    def test_should_run_a_replication_and_not_repeat_it(self, package: Path, tmp_path: Path):
        # Act
        first = self._run(package, tmp_path, "bash", "train.sbatch", TASK="2")
        again = self._run(package, tmp_path, "bash", "train.sbatch", TASK="2")

        # Assert
        assert first.returncode == 0, first.stderr
        assert (package / "results" / "run02" / "results.yaml").is_file()
        assert "already done" in again.stdout

    def test_should_bring_back_a_replication_from_the_scratch_disk_even_if_it_fails(
        self, package: Path, tmp_path: Path
    ):
        # Act
        scratch = tmp_path / "scratch"
        result = self._run(
            package, tmp_path, "bash", "train.sbatch", TASK="1", SCRATCH_DIR=str(scratch), FAIL="1"
        )

        # Assert
        assert result.returncode == 3
        assert (package / "results" / "run01" / "results.yaml").is_file()
        assert not scratch.exists()

    def test_should_submit_only_the_replications_not_done(self, package: Path, tmp_path: Path):
        # Arrange
        self._run(package, tmp_path, "bash", "train.sbatch", TASK="2")

        # Act
        result = self._run(package, tmp_path, "./submit.sh", "--qos=short")

        # Assert
        assert result.returncode == 0, result.stderr
        assert "sbatch --array=1,3 --qos=short train.sbatch" in result.stdout


def _finish(package: Path, replication: str, state: str = "FINISHED") -> None:
    """Leave a replication as Evolver does: its results, status and pointer to them."""
    output = package / "results" / replication
    output.mkdir(parents=True)
    (output / "VAR_CONF.txt").write_text(
        "# Evaluation: 40\nEpsilon=0.1 NormalizedHypervolume=0.2 | --neighborhoodSize 20\n"
    )
    (output / "METADATA.txt").write_text("Meta-evaluations performed: 40\n")
    (output / "status.yaml").write_text(
        yaml.safe_dump(
            {
                "status": state,
                "evaluationsDone": 40,
                "maxEvaluations": 40,
                "updatedAt": "2026-11-26T12:00:00",
            }
        )
    )
    (output / "results.yaml").write_text(
        f"outputDirectory: results/{replication}\n"
        f"metadataFile: results/{replication}/METADATA.txt\n"
    )


class TestImportClusterResults:
    @pytest.fixture
    def package(self, tmp_path: Path) -> Path:
        package = _unzip(_write(tmp_path, replications=3), tmp_path / "cluster")
        _finish(package, "run01")
        _finish(package, "run02", state="FAILED")
        (package / "logs").mkdir()
        (package / "logs" / "evolver_tutorial_77_1.out").write_text("Replication run01\n")
        return package

    def test_should_make_each_finished_replication_a_training_run(
        self, package: Path, tmp_path: Path
    ):
        # Act
        result = import_cluster_results(package, tmp_path / "studio")

        # Assert
        assert isinstance(result, Ok)
        assert result.value.imported == (f"{RUN_ID}-run01",)
        assert any("run02" in line and "FAILED" in line for line in result.value.skipped)
        (training,) = list_finished_trainings(tmp_path / "studio")
        assert training.run_id == f"{RUN_ID}-run01"
        assert training.experiment == "tutorial"
        assert training.algorithm == "MOEAD"
        assert training.output_directory == (
            tmp_path / "studio" / "results" / "MOEAD" / f"tutorial.{RUN_ID}-run01"
        )
        assert training.parameter_space_file.read_text() == "algorithmResult: {}\n"
        assert training.reference_fronts == ("resources/referenceFronts/ZDT1.csv",)
        run_dir = training.run_dir
        assert (run_dir / "runner.log").read_text() == "Replication run01\n"
        pointer = yaml.safe_load((run_dir / "results.yaml").read_text())
        assert pointer["metadataFile"] == f"results/MOEAD/tutorial.{RUN_ID}-run01/METADATA.txt"

    def test_should_import_only_what_is_new_the_next_time(self, package: Path, tmp_path: Path):
        # Arrange
        import_cluster_results(package, tmp_path / "studio")
        _finish(package, "run03")

        # Act
        result = import_cluster_results(package, tmp_path / "studio")

        # Assert
        assert result.value.imported == (f"{RUN_ID}-run03",)
        assert any("run01: already imported" in line for line in result.value.skipped)

    def test_should_give_back_a_front_the_package_had_moved(self, tmp_path: Path):
        # Arrange
        front = tmp_path / "elsewhere" / "Mine.csv"
        front.parent.mkdir()
        front.write_text("0,1\n")
        package = _unzip(_write(tmp_path, _package(str(front))), tmp_path / "cluster")
        _finish(package, "run01")

        # Act
        import_cluster_results(package, tmp_path / "studio")

        # Assert
        (training,) = list_finished_trainings(tmp_path / "studio")
        assert training.reference_fronts == (str(front),)

    def test_should_refuse_a_folder_that_is_not_a_package(self, tmp_path: Path):
        result = import_cluster_results(tmp_path, tmp_path / "studio")

        assert isinstance(result, Err)
        assert MANIFEST_FILE_NAME in result.message

    def test_should_say_when_no_replication_has_run(self, tmp_path: Path):
        package = _unzip(_write(tmp_path), tmp_path / "cluster")

        result = import_cluster_results(package, tmp_path / "studio")

        assert isinstance(result, Err)
        assert "no replication" in result.message
