"""Runs the jobs of a validation study in parallel: `python -m evolver_studio.validation_worker`.

The Streamlit page launches it as a detached process (see `validation_runner`), so a study that
takes a long while goes on when the page is closed. It reads the study's manifest, runs the
`SolveRunnerMain` of each job with a pool of processes, and keeps the study's `status.yaml` in the
format Evolver's own runs use (`evaluationsDone` and `maxEvaluations` count the jobs), so that
`runs.run_phase` and `progress.running_label` work on a study as they do on a run.
"""

import datetime as dt
import signal
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from evolver_studio.evolver_client import SOLVE_RUNNER_MAIN_CLASS, WORKING_DIRECTORY
from evolver_studio.validation import JOBS_DIRECTORY_NAME, read_manifest

JOB_LOG_NAME = "runner.log"
FAILED_JOBS_NAME = "failed_jobs.txt"


class StudyRunner:
    """Runs the jobs of one study.

    Attributes:
        study_directory: The study's directory (validation-runs/<id>/).
        jar: Evolver's jar.
        processes: How many jobs run at the same time.
        working_directory: The JVM's working directory.
    """

    def __init__(
        self,
        study_directory: Path,
        jar: Path,
        processes: int,
        working_directory: Path = WORKING_DIRECTORY,
    ) -> None:
        self.study_directory = study_directory
        self.jar = jar
        self.processes = max(processes, 1)
        self.working_directory = working_directory
        self._lock = threading.Lock()
        self._children: set[subprocess.Popen] = set()
        self._stopping = False
        self._finished = 0
        self._failed: list[int] = []
        self._total = 0

    def run(self) -> bool:
        """Run every job of the study.

        Returns:
            Whether every job succeeded; false when one failed or the study was stopped.
        """
        manifest = read_manifest(self.study_directory)
        if manifest is None:
            return False
        jobs = manifest["jobs"]
        self._total = len(jobs)
        self._write_status("RUNNING")
        with ThreadPoolExecutor(max_workers=self.processes) as pool:
            list(pool.map(self._run_job, jobs))
        self._write_failed_jobs()
        if self._stopping:
            return False
        self._write_status("FINISHED")
        return not self._failed

    def stop(self) -> None:
        """Stop the study: no new job starts and the ones running are terminated."""
        self._stopping = True
        with self._lock:
            for child in self._children:
                child.terminate()

    def _run_job(self, job: dict) -> None:
        if self._stopping:
            return
        directory = self.study_directory / JOBS_DIRECTORY_NAME / job["directory"]
        command = [
            "java",
            "-cp",
            str(self.jar),
            SOLVE_RUNNER_MAIN_CLASS,
            str(directory / "request.yaml"),
            str(directory / "status.yaml"),
        ]
        succeeded = False
        try:
            with (directory / JOB_LOG_NAME).open("w") as log:
                child = subprocess.Popen(
                    command,
                    cwd=self.working_directory,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                with self._lock:
                    self._children.add(child)
                succeeded = child.wait() == 0
                with self._lock:
                    self._children.discard(child)
        except OSError:
            succeeded = False
        with self._lock:
            if self._stopping:
                return
            self._finished += 1
            if not succeeded:
                self._failed.append(job["number"])
            self._write_status("RUNNING")

    def _write_status(self, state: str) -> None:
        status = {
            "status": state,
            "evaluationsDone": self._finished,
            "maxEvaluations": self._total,
            "updatedAt": dt.datetime.now().isoformat(),
        }
        target = self.study_directory / "status.yaml"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(yaml.safe_dump(status))
        temporary.replace(target)

    def _write_failed_jobs(self) -> None:
        if self._failed:
            numbers = "\n".join(str(number) for number in sorted(self._failed))
            (self.study_directory / FAILED_JOBS_NAME).write_text(numbers + "\n")


def main(arguments: list[str]) -> int:
    """Run a study: `<study directory> <jar> <processes> [<working directory>]`."""
    study_directory, jar, processes = Path(arguments[0]), Path(arguments[1]), int(arguments[2])
    working_directory = Path(arguments[3]) if len(arguments) > 3 else WORKING_DIRECTORY
    runner = StudyRunner(study_directory, jar, processes, working_directory)

    def stop(signal_number, frame) -> None:
        runner.stop()

    signal.signal(signal.SIGTERM, stop)
    return 0 if runner.run() else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
