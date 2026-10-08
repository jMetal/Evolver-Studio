# Changelog

All notable changes to Evolver-Studio are documented in this file. Evolver-Studio has its own
version numbers, independent of Evolver's: the [README](README.md) says which Evolver release each
version works with.

## 0.1.0 (unreleased)

Works with Evolver 2.4.

### Moving to Evolver 2.4 (on `experiment/evolver-3-sync`)

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
  per problem and every run, with CSV downloads. Studies are kept under `validation-runs/` and can
  be reopened. Needs scipy.
- **Tutorial S3, *Validating a configuration*** (pairs with E8): it runs a small validation study
  in the background and reads it step by step. The pivot is the NSGA-II that Evolver's tutorial E8
  tuned for the nine bi-objective WFG problems (bundled in Evolver's jar), against the default
  NSGA-II, MOEA/D and RVEA, on two WFG problems it was tuned for and on ZDT1 and DTLZ2 with two
  objectives, which it never saw. The steps read the medians, the Wilcoxon test and the A12 effect
  size, what the same comparison says with 5 to 15 runs, and the problems seen and not seen in the
  tuning; what the text says about the results is read from them, and one comparison that the test
  cannot settle is used as the example. "Open in Validation" fills Validation's form with the study.
  It needs Evolver 2.4 or later.
- **Tutorials** numbered as Evolver numbers its own: only the written ones have a number,
  consecutive by level (*Exploring a parameter space* is now S1 and *Solving a problem with a
  configurable algorithm* S2), and the planned ones are listed without one. Each pairs with
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
  with the flat encoding (the tree encoding is browsable in Explore but cannot be launched yet).
  Arbitrary multi-problem training sets, and a guided or expert editor of the base algorithm's
  parameter space.
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
