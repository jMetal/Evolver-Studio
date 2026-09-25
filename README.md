<p align="center">
  <img src="assets/logo/studio-logo.svg" alt="Evolver-Studio" width="480">
</p>

# Evolver-Studio: a graphical interface for Evolver

Evolver-Studio is a Python/[Streamlit](https://streamlit.io/) application for
[Evolver](https://github.com/jMetal/Evolver), the Java framework for the automated
meta-optimization of multi-objective metaheuristics. Evolver has no GUI: its algorithms are
configured through YAML parameter spaces, and training runs are launched from Java code or its
command-line entry point. Evolver-Studio makes Evolver usable without writing Java or editing YAML
by hand, and adds an analysis layer on top of its results.

It serves two purposes:

- **Solving problems**: configure and run Evolver's configurable algorithms on a concrete problem,
  in the style of jMetal's runners. Evolver's configurable core can be used on its own, as an
  alternative to jMetal.
- **Meta-optimization**: launch and monitor training runs, then analyze and validate the
  configurations they find.

> **Status:** early development (v0.1.0). Exploring parameter spaces, launching and monitoring
> training runs, and the first tutorial are available; the solving, analysis and validation pages
> are planned. See [ROADMAP.md](ROADMAP.md).

## How it works

Evolver-Studio drives Evolver as a subprocess through a file-based contract, rather than a server:

```text
Evolver-Studio (Streamlit)
  └─> writes request.yaml (+ reusable baseLevel / metaSearch files)
       └─> java -cp Evolver-2.1-jar-with-dependencies.jar
                org.uma.evolver.cli.training.TrainingRunnerMain request.yaml status.yaml
            ├─> status.yaml   polled for progress (RUNNING / FINISHED / FAILED)
            └─> METADATA.txt, INDICATORS.csv, CONFIGURATIONS.csv, VAR_CONF.txt
```

Training runs are long batch jobs, so each one runs as a detached process: closing or reloading the
browser does not stop it, and reopening the app reconnects to it. The catalogue of algorithms,
problems and indicators is cross-checked against the manifest printed by Evolver's
`org.uma.evolver.cli.training.DescribeMain`.

## Features

The app's menu groups its pages by purpose:

| Section | Page | Status | What it does |
|---|---|:---:|---|
| Explore | Explore | ✅ | Browse any algorithm's parameter space as a tree; see each meta-optimizer's encodings and operator parameters |
| Solve | Run algorithm | 🚧 | Configure and run an algorithm on a problem, inspect its front, export `VAR`/`FUN` |
| Meta-optimization | Training | ✅ | Configure, launch, monitor and cancel a training run |
| Meta-optimization | Analysis | 🚧 | Statistical comparison of runs and configurations (Wilcoxon tests, tables, plots) |
| Meta-optimization | Validation | 🚧 | Run a tuned configuration on a validation set and compare it with the default configuration |
| Learn | Tutorials | ✅ | Interactive, step-by-step tutorials that pair with Evolver's documentation |

✅ available · 🚧 planned

The Training page provides:

- A parameter editor for the base-level algorithm, with a **guided** mode (forms, valid by
  construction) and an **expert** mode (raw YAML, validated on every change).
- Training sets of one or more problems, each with its reference front and evaluation budget.
  Problems can be any jMetal problem, given by class name.
- A choice of meta-optimizer, with its operator settings pre-filled from Evolver's example files.
- A live plot of the indicator front as the run progresses.

## Supported algorithms

Evolver-Studio can browse the parameter space of every configurable algorithm in Evolver. The ones
that can be launched from the app today are:

| Level | Algorithms |
|---|---|
| Base level | NSGA-II (Double, Permutation), MOEA/D (Double) |
| Meta level | NSGA-II, AGE-MOEA, SPEA2, SMPSO, Async NSGA-II, Random Search (flat encoding) |

The other base-level algorithms (SMS-EMOA, RDE-MOEA, AGE-MOEA, RVEA, MOPSO, NSGA-III, PAES,
SSMOEA) are browsable only, until Evolver's command-line runner supports them. See Evolver's
[supported algorithms](https://github.com/jMetal/Evolver#supported-algorithms) for the full list.

## Requirements

- Python 3.11+ and [Conda](https://docs.conda.io/)
- Java 21 (JDK 21 recommended) and Maven 3.6+, to build Evolver
- **Evolver 2.1** (the `v2.1` tag), the first release that includes the `cli.training` command-line
  runner. Other versions are not supported; the expected version is set by `EVOLVER_VERSION` in
  `evolver_studio/evolver_client.py`.

## Installation

Build Evolver 2.1:

```bash
git clone --branch v2.1 https://github.com/jMetal/Evolver.git
cd Evolver
mvn clean package -DskipTests   # builds target/Evolver-2.1-jar-with-dependencies.jar
cd ..
```

Install Evolver-Studio in its own Conda environment:

```bash
git clone https://github.com/jMetal/Evolver-Studio.git
cd Evolver-Studio
make env   # creates (or updates) the 'evolver-studio' environment from environment.yml
```

## Quick start

```bash
make run   # streamlit run app.py
```

1. In the sidebar, set **Evolver checkout path** to the directory where you cloned Evolver. The
   **Build Evolver** button builds the jar from there if you have not built it yet.
2. Open **Explore** to browse the algorithms and their parameter spaces.
3. Open **Training**, keep the default settings (NSGA-II on ZDT4) or edit them, and click
   **Launch training**. The indicator front updates as the run progresses.
4. Paths are relative to the Evolver checkout. Each run's request files (`request.yaml`,
   `status.yaml`, ...) are kept under `cli-runner-runs/<run_id>/`, and its results under
   `<output directory>/<run_id>/` (by default, `results/nsgaii/ZDT4/<run_id>/`).

New to Evolver? Start with the **Tutorials** page.

## Development

```bash
make lint     # ruff check .
make format   # ruff format .
make test     # pytest tests/ -x
```

Tests that call Evolver's jar run only when the jar is found in the Evolver checkout; they are
skipped otherwise.

Evolver adds algorithms and parameter spaces over time, and Evolver-Studio's catalogue
(`evolver_studio/catalogue.py`) is maintained by hand. To keep them in sync, `tests/test_catalogue.py`
scans Evolver's parameter-space directory and fails on any file the catalogue does not account for;
Evolver has matching tests for its algorithm and meta-optimizer registries.

Code and commits follow [CODING_GUIDELINES.md](CODING_GUIDELINES.md) and
[GIT_GUIDELINES.md](GIT_GUIDELINES.md).

## Roadmap

Next steps include tree-encoding training runs, the solving track (which needs a new Evolver entry
point for single algorithm runs), the analysis layer and validation runs. See
[ROADMAP.md](ROADMAP.md) for details and the use cases behind them.

## Citation

Evolver-Studio has no publication of its own yet. If you use it in your research, please cite
Evolver:

```bibtex
@article{AND23,
  title   = {Evolver: Meta-optimizing multi-objective metaheuristics},
  journal = {SoftwareX},
  volume  = {23},
  pages   = {101551},
  year    = {2024},
  issn    = {2352-7110},
  doi     = {10.1016/j.softx.2023.101551},
}
```

## License

This project is licensed under the GNU General Public License v3.0 — see the [LICENSE](LICENSE)
file for details.
