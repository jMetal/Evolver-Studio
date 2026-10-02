# Changelog

All notable changes to Evolver-Studio are documented in this file. Evolver-Studio has its own
version numbers, independent of Evolver's: the [README](README.md) says which Evolver release each
version works with.

## 0.1.0 (unreleased)

Works with Evolver 2.2.

### Added

- **Explore**, four read-only pages: *Base algorithms* and *Meta-optimizers* show the parameter space
  of each algorithm, per encoding, as a filterable table (one row per parameter, with the condition
  that activates it); *Quality indicators* lists the indicators a training run can minimize;
  *Problems* is a placeholder. The pages open with nothing selected, and warn when the Evolver jar
  in use is older than the release the app mirrors.
- **Training**: configure, launch, monitor and cancel a meta-optimization run, with a live view of
  the front of configurations found. NSGA-II (Double and Permutation), MOEA/D and RVEA as base
  algorithms; NSGA-II, AGE-MOEA, SPEA2, SMPSO, AsyncNSGA-II and Random Search as meta-optimizers,
  with the flat encoding and, for NSGA-II, AGE-MOEA, AsyncNSGA-II and Random Search, the tree
  encoding. Arbitrary multi-problem training sets, and a guided or expert editor of the base
  algorithm's parameter space.
- **Tutorials**, with the first one, *Exploring a parameter space*, which pairs with Evolver's
  tutorial E1.
- A **home page** with a card for each part of the app, and a menu grouped by purpose (Explore,
  Solve, Meta-optimization, Learn). *Run algorithm*, *Analysis* and *Validation* are placeholders.
- No Evolver checkout and no Maven needed: the app downloads Evolver's release jar from Maven Central
  (checksum-verified) and copies the reference fronts, weight vectors and TSP instances it needs from
  the release's tag (`make sync-resources`). `EVOLVER_JAR` runs it against another build.
- Drift detection between the app's catalogue of algorithms and Evolver's `DescribeMain` manifest and
  parameter spaces (`tests/test_catalogue.py`), and the `/sync-catalogue` and `/bump-evolver`
  commands to keep the app in line with an Evolver release.
