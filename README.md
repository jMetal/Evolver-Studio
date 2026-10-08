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

> **Status:** early development, heading to 0.1.0 (see [CHANGELOG.md](CHANGELOG.md)). Exploring
> parameter spaces, launching and monitoring training runs, running an algorithm on a problem,
> validating a tuned configuration, and three tutorials are available; the analysis page is planned.
> See [ROADMAP.md](ROADMAP.md).

## How it works

Evolver-Studio runs Evolver's release jar, downloaded from Maven Central, as a subprocess, and
talks to it through files rather than a server:

```text
Evolver-Studio (Streamlit)
  └─> writes request.yaml (+ reusable baseLevel / metaSearch files)
       └─> java -cp Evolver-2.4-jar-with-dependencies.jar
                org.uma.evolver.cli.training.TrainingRunnerMain request.yaml status.yaml
            ├─> status.yaml   polled for progress (RUNNING / FINISHED / FAILED)
            └─> METADATA.txt, INDICATORS.csv, CONFIGURATIONS.csv, VAR_CONF.txt
```

Training runs are long batch jobs, so each one runs as a detached process: closing or reloading the
browser does not stop it, and reopening the app reconnects to it. The parameter spaces the app
shows and edits are read from the jar itself, and the reference fronts and weight vectors that
training runs need are copied from Evolver into [`resources/`](resources/README.md). The catalogue
of algorithms,
problems and indicators is cross-checked against the manifest printed by Evolver's
`org.uma.evolver.cli.training.DescribeMain`.

## Features

The app opens on a home page with a card for each part of the tool. The menu groups the pages by
purpose:

| Section | Page | Status | What it does |
|---|---|:---:|---|
| Explore | Base algorithms | ✅ | Browse an algorithm's parameter space, per encoding, as a filterable table (one row per parameter, with the condition that activates it) |
| Explore | Meta-optimizers | ✅ | See each meta-optimizer's encodings and the operators it can be configured with |
| Explore | Quality indicators | ✅ | The indicators a training run can minimize, and what each one measures |
| Explore | Problems | ✅ | The problems available for training and solving: encoding, objectives, variables, arguments and reference fronts |
| Solve | Run algorithm | ✅ | Run an algorithm on a problem from its default configuration, adjusted within its parameter space; inspect the fronts and indicators, download `VAR`/`FUN` |
| Meta-optimization | Training | ✅ | Configure, launch, monitor and cancel a training run |
| Meta-optimization | Analysis | 🚧 | Statistical comparison of runs and configurations (Wilcoxon tests, tables, plots) |
| Meta-optimization | Validation | ✅ | Compare a tuned configuration with the default configurations of other algorithms on a set of problems: many runs, medians, Wilcoxon tests, effect sizes and boxplots |
| Learn | Tutorials | ✅ | Interactive, step-by-step tutorials that pair with Evolver's documentation |

✅ available · 🚧 planned

The Training page provides:

- A parameter editor for the base-level algorithm, with a **guided** mode (forms, valid by
  construction) and an **expert** mode (raw YAML, validated on every change).
- Training sets of one or more problems, each with its reference front and evaluation budget.
  The problems are chosen from a list (those of the base algorithm's encoding), or from the same
  listing as Explore › Problems, with each problem's dimensions, arguments and fronts in view; any
  other jMetal problem can be added by class name. A problem can be given its constructor's
  arguments (e.g. DTLZ2 with 2 objectives), and one of another encoding than the base algorithm's
  is reported before launching.
- A choice of meta-optimizer, with its operator settings pre-filled from Evolver's example files.
- A live plot of the indicator front as the run progresses.

## Supported algorithms

Evolver-Studio can browse the parameter space of every configurable algorithm in Evolver. The ones
that can be launched from the app today are:

| Level | Algorithms |
|---|---|
| Base level | NSGA-II, MOEA/D, SMS-EMOA, PAES (Double, Binary, Permutation); RDE-MOEA (Double, Permutation); NSGA-III, AGE-MOEA, RVEA, SSMOEA (Double) |
| Meta level | NSGA-II, AGE-MOEA, SPEA2, SMPSO, Async NSGA-II, Random Search (flat encoding); NSGA-II, AGE-MOEA, Async NSGA-II, Random Search (tree encoding) |

MOPSO is browsable only, until Evolver's command-line runner supports it. See Evolver's
[supported algorithms](https://github.com/jMetal/Evolver#supported-algorithms) for the full list.

## Requirements

- Python 3.11+ and [Conda](https://docs.conda.io/)
- Java 21 or newer, on the `PATH`

Neither an Evolver checkout nor Maven is needed: the app downloads the jar of **Evolver 2.4** (the
release the app is built against) from Maven Central. Other
versions are not supported; the expected version is set by `EVOLVER_VERSION` in
`evolver_studio/evolver_client.py`.

Evolver-Studio is versioned on its own, and works with one Evolver release at a time:

| Evolver-Studio | Evolver |
|---|---|
| 0.1.0 (in development) | 2.4 |

## Installation

```bash
git clone https://github.com/jMetal/Evolver-Studio.git
cd Evolver-Studio
make env   # creates (or updates) the 'evolver-studio' environment from environment.yml
```

## Quick start

```bash
make run   # streamlit run app.py
```

1. The first time, click **Download Evolver 2.4** in the sidebar. The jar (about 130 MB) is
   downloaded from Maven Central into `lib/` and its checksum is verified.
2. The home page shows the parts of the app. Open **Base algorithms**, under Explore, to browse the algorithms and their parameter spaces.
3. Open **Training**, keep the default settings (NSGA-II on ZDT4) or edit them, and click
   **Launch training**. The indicator front updates as the run progresses.
4. Paths are relative to the Evolver-Studio directory. Each run's request files (`request.yaml`,
   `status.yaml`, ...) are kept under `cli-runner-runs/<run_id>/`, and its results under
   `<output directory>/<run_id>/` (by default, `results/nsgaii/ZDT4/<run_id>/`).

New to Evolver? Start with the **Tutorials** page.

## Development

```bash
make lint     # ruff check .
make format   # ruff format .
make test     # pytest tests/ -x
```

To run the app against another Evolver build, such as one made from Evolver's `develop` branch
while working on Evolver itself, point the `EVOLVER_JAR` environment variable at its jar:

```bash
EVOLVER_JAR=/path/to/Evolver-2.5-SNAPSHOT-jar-with-dependencies.jar make run
```

Tests that use Evolver's jar (downloaded, or set by `EVOLVER_JAR`) are skipped without it. Among
them, `tests/test_catalogue.py` keeps Evolver-Studio's hand-maintained catalogue
(`evolver_studio/catalogue.py`) in sync with Evolver: it fails on any parameter space in the jar
that the catalogue does not account for, and cross-checks the catalogue against the manifest
printed by Evolver's `DescribeMain`. Evolver has matching tests for its algorithm and
meta-optimizer registries.

Moving to a new Evolver release means changing `EVOLVER_VERSION` in
`evolver_studio/evolver_client.py` and refreshing the copied resources with
`make sync-resources`, which downloads them from the release's source archive on GitHub.
`tests/test_resources.py` checks the copy against the checksums that command records.

Code and commits follow [CODING_GUIDELINES.md](CODING_GUIDELINES.md) and
[GIT_GUIDELINES.md](GIT_GUIDELINES.md).

## Roadmap

Next steps include the analysis layer, and more statistics for
validation studies (Friedman and critical difference plots). See
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
