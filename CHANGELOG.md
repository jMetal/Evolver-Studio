# Changelog

All notable changes to Evolver-Studio are documented in this file. Evolver-Studio has its own
version numbers, independent of Evolver's: the [README](README.md) says which Evolver release each
version works with.

## 0.1.0 (unreleased)

Works with Evolver 2.4.

### Tutorials

- **Tutorial S1, *A tour of Evolver-Studio*** (pairs with E4): how the app connects to Evolver (Java
  and the jar, checked live), what Evolver offers (read from the jar's manifest), what each page is
  for, and where runs and results are kept.
- **Tutorial S4, *Analyzing training results*** (pairs with E7): reads a real training that ships
  with the app (`evolver_studio/tutorial_data/analysis/`) with the views of the Training analysis page —
  summary, convergence, front and population, final-front configurations, parameter agreement.
- The written tutorials are renumbered, as Evolver does: S1 tour, S2 parameter spaces, S3 solving,
  S4 analysis, S5 validation.

### Quality indicators: which way is better

- Each quality indicator of the catalogue says whether it is **maximized**, and the analysis of a
  validation follows it (the best median, the A12 effect size and the Wilcoxon verdict, the
  highlighted cells, "lower/higher is better", the `Maximize` column of SAES's metrics file).
  Every indicator Evolver offers today is minimized; Explore › Quality indicators shows the
  direction of each.
- *Validation* no longer offers **HypervolumeMinus**: it only exists so that a training can
  minimize the hypervolume; *NormalizedHypervolume* tells the same about a validation.

### The four steps of meta-optimization

- The Meta-optimization menu follows the workflow: **Training**, **Training analysis** (the page
  called *Analysis* until now), **Validation** and **Validation analysis**, which is still to come
  (Validation shows a study's results meanwhile).

### Saving tables and charts

- *Validation* saves a study's results as the two CSV files **SAES** reads: the results (one row
  per run and indicator: `Algorithm`, `Instance`, `MetricName`, `ExecutionId`, `MetricValue`) and
  the metrics (`MetricName`, `Maximize`), in its *Runs* tab.
- With SAES installed (`pip install SAES`), *Validation* saves the comparison with the pivot as
  **SAES's Wilcoxon pivot table** in LaTeX, as Evolver's `scripts/wilcoxon_pivot_tables.py` makes
  it: median and interquartile range, the pivot in the last column, `+`/`-`/`=` against it and a
  last row that counts them. The counts are written whole in bold and the preamble is one that
  compiles, two fixes to SAES's document that the script also makes.
- **Every chart** of the app (fronts, solutions, boxplots, convergence, population, parameters, in
  the pages and in the tutorials) has **PNG** and **PDF** buttons. The images are made with Kaleido
  (a new dependency) in the Chrome installed on the computer, only when a button is pressed.

### Validation

- The algorithms the tuned configuration is compared with start from their default configuration,
  and each has an **Adjust the parameters** form, the one of *Run algorithm*: a changed algorithm
  enters the study as "<name> (adjusted)", and *Open in Validation* brings it back adjusted.

### Training analysis

- *Training*'s best configurations, while it runs and once it has finished, have a **Validate this
  configuration** button that opens *Validation* with the chosen one as the tuned configuration (the
  box can still be copied and pasted there by hand). Validation opens with the algorithm and the
  population size of the training, and the list of problems empty, for you to fill. The
  indicators are Validation's own, not those the training minimized: a validation measures every
  front it gets with the indicators chosen for it.

- *Meta-optimization › Training analysis* lists the finished trainings and, for the one chosen, shows how it
  was run and what it took, how its meta-objectives converged, the front and (if it was written) the
  population at each checkpoint, and the configurations of its final front. A configuration can be
  downloaded, sent to *Validation* as the tuned configuration, or to *Run algorithm* on one of the
  training's problems. A *Parameters* tab shows, per parameter, how much the front's configurations
  agree and where they differ from the algorithm's default configuration.

### Moving to Evolver 2.4

- **Base algorithms**: NSGA-III, SMS-EMOA, RDE-MOEA, AGE-MOEA, PAES and SSMOEA can be launched from
  Training and Run algorithm, and MOEA/D also with its Binary and Permutation encodings, as
  Evolver's `DescribeMain` registers them. Run algorithm starts from the default configuration the
  jar ships for each, where there is one. MOPSO is still browsable only.
- **Problems**, from the problem catalogue of Evolver's `DescribeMain`:
  - *Explore › Problems* lists every problem with its family, encoding, number of objectives and
    variables, constructor arguments and reference fronts, filtered by encoding, family and name.
  - *Run algorithm* offers binary and permutation problems (ZDT5, OneZeroMax, the TSP instances),
    shows only the algorithms that solve the chosen problem's encoding, lets its arguments be set
    (the number of objectives and variables of DTLZ, WFG or ZCAT, for instance), picks the
    reference front with its number of objectives and warns when the chosen one has another. A
    past run's arguments are restored with it.
  - *Training*: a problem of the training set can be given its arguments, the curated names
    offered are those of the base algorithm's encoding, and a problem of another encoding is
    reported before launching.
- **Validation**: compare a tuned configuration (a configuration of a finished training run, or
  one pasted) with the default configurations of Evolver's algorithms, on problems of one encoding.
  Every algorithm runs many times on every problem, with the same population size, evaluation
  budget and seeds, in the background (a blinking label counts the jobs finished, and the study can
  be cancelled); an algorithm that reads weight vectors is left out of the problems that have no
  file for the population size. The results show the median of each indicator per problem, a
  Wilcoxon rank-sum test and the A12 effect size of each algorithm against the tuned one, boxplots
  per problem and every run, with CSV downloads. The problems are chosen from a list or from the
  listing of Explore › Problems, as in Training. Studies are kept under `validation-runs/` and can
  be reopened. Needs scipy.
- **Tutorial S5, *Validating a configuration**** (pairs with E8): it runs a small validation study
  in the background and reads it step by step. The pivot is the NSGA-II that Evolver's tutorial E8
  tuned for the nine bi-objective WFG problems (bundled in Evolver's jar), against the default
  NSGA-II, MOEA/D and RVEA, on two WFG problems it was tuned for and on ZDT1 and DTLZ2 with two
  objectives, which it never saw. The steps read the medians, the Wilcoxon test and the A12 effect
  size, what the same comparison says with 5 to 15 runs, and the problems seen and not seen in the
  tuning; what the text says about the results is read from them, and one comparison that the test
  cannot settle is used as the example. "Open in Validation" fills Validation's form with the study.
  It needs Evolver 2.4 or later.
- **Training monitor**, for runs that last hours: tabs with the progress, pace, time left and a
  warning when the run seems stuck (*Overview*); the front; optionally the meta-optimizer's whole
  population at a checkpoint, with the front over it and a control to browse the earlier ones
  (*Population*, from a checkbox before launching, which makes Evolver write
  `POPULATION_INDICATORS.csv`); how the best, the median and the worst of each meta-objective
  improve, by evaluations or by computing time (*Convergence*); the configurations found so far,
  one of them downloadable (*Best so far*); and the runner's log. The files are followed from where
  the last refresh stopped, so a run of days does not slow the page down, and a finished run shows
  the same views. Verified with real runs.
- **Training**: the size of the meta-optimizer's population can be chosen (Random Search has none),
  the update frequency must be a multiple of it (Evolver writes a checkpoint only then, and a
  frequency that was not gave no checkpoints), SPEA2 and SMPSO say which operators they hardcode
  and why they have no tree encoding instead of showing an empty box, and the preview of the
  request says where the parameter space file will be instead of a placeholder.
- **Fixes in Training**: the runner's output goes to `runner.log` (it was a pipe that nobody read,
  which in a long run could block Evolver); a run that has not written its first status yet is
  found, so the page no longer shows the launch form again right after *Launch*; and the
  configurations a training found are those of its last checkpoint, no longer those of every
  checkpoint (this affected the tuned configurations Validation offers).
- **Training**: a selector chooses the meta-optimizer's encoding, flat or tree (the latter for the
  meta-optimizers that support it), right before the box with its operator flags, which start
  from the example file of the chosen encoding; and another chooses when the meta-optimizer stops,
  after a number of evaluations or an amount of computing time (Evolver takes one limit or the
  other), with the progress shown as time spent of the limit. Verified with real runs of the tree
  encoding and of a time limit.
- **Training**: the problems of the training set are chosen from a list instead of typed (the
  problems of the base algorithm's encoding, with the option to type any other jMetal class name),
  and a *Browse the problems* switch shows the listing of Explore › Problems, with its filters, to
  select several rows at once and add them to the set. Each chosen problem gets its reference
  front when there is no doubt about which one (DTLZ2's three-objective front, for instance).
- **Tutorials** numbered as Evolver numbers its own: only the written ones have a number,
  consecutive by level (*Exploring a parameter space* is now S2 and *Solving a problem with a
  configurable algorithm* S3), and the planned ones are listed without one. Each pairs with
  Evolver's tutorials by their new numbers, and the catalogue links to Evolver's tutorials index.

### Added

- **Explore**, four read-only pages: *Base algorithms* and *Meta-optimizers* show the parameter space
  of each algorithm, per encoding, as a filterable table (one row per parameter, with the condition
  that activates it); *Quality indicators* lists the indicators a training run can minimize;
  *Problems* is a placeholder. The pages open with nothing selected, and warn when the Evolver jar
  in use is older than the release the app mirrors.
- **Training**: configure, launch, monitor and cancel a meta-optimization run, with a live view of
  the front of configurations found. NSGA-II (Double, Binary and Permutation), MOEA/D and RVEA as base
  algorithms; NSGA-II, AGE-MOEA, SPEA2, SMPSO, AsyncNSGA-II and Random Search as meta-optimizers,
  with the flat encoding and, for NSGA-II, AGE-MOEA, AsyncNSGA-II and Random Search, the tree
  encoding. Arbitrary multi-problem training sets, and a guided or expert editor of the base
  algorithm's parameter space.
- **Run algorithm**: run one of Evolver's configurable algorithms (NSGA-II, MOEA/D, RVEA) on a
  problem, through `cli.solving`. The configuration starts from the algorithm's default one and can
  be adjusted within its parameter space (a widget per active parameter, limited to what the space
  allows, with the parameters that changed marked); the fronts of the independent runs are plotted
  over the problem's reference front, with the quality indicators of each run and their summary, and
  the results can be downloaded. Every run is kept under `solve-runs/` and can be reopened.
  For MOEA/D and RVEA the population size is chosen among the sizes that have a weight vector file
  for the problem's number of objectives, instead of failing when none matches. A past run can be
  repeated: "Use this configuration for a new run" fills the form with its problem, algorithm,
  configuration (the adjusted one included) and budget, and "Download request.yaml" gives the file
  to run it outside Evolver-Studio with `SolveRunnerMain`. A *Solutions* tab lists the solutions of
  a run with their objectives and decision variables, filters them by a range for each objective,
  shows the variables of a selected one (as a permutation when it is one) and downloads the table.
  While a run is in progress the page shows a blinking label with the evaluation it is at (with
  Evolver 2.3 or later; with an older one, only that it is running) and a button to cancel it, and
  moves on to its results when it ends.
- **Tutorials**, with the first one, *Exploring a parameter space*, which pairs with Evolver's
  tutorial E1: it reads a space in the same table the Explore pages show, with its *Active if*
  column and filter, writes the configuration a choice of values gives, and ends with the
  meta-optimizers' operators and the quality indicators. The second one, *Solving a problem with a
  configurable algorithm* (pairs with E2), runs NSGA-II, MOEA/D and RVEA through Run algorithm:
  every step has a prepared run (fixed seed, with the values the text announces) that can be run in
  the tutorial or opened in Run algorithm, to repeat it by hand.
- A **home page** with a card for each part of the app, and a menu grouped by purpose (Explore,
  Solve, Meta-optimization, Learn). *Analysis* is a placeholder.
- No Evolver checkout and no Maven needed: the app downloads Evolver's release jar from Maven Central
  (checksum-verified) and copies the reference fronts, weight vectors and TSP instances it needs from
  the release's tag (`make sync-resources`). `EVOLVER_JAR` runs it against another build.
- Drift detection between the app's catalogue of algorithms and Evolver's `DescribeMain` manifest and
  parameter spaces (`tests/test_catalogue.py`), and the `/sync-catalogue` and `/bump-evolver`
  commands to keep the app in line with an Evolver release.
