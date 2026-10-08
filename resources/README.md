# Resources copied from Evolver

The files in this directory are copied, unchanged, from the `resources/` directory of
[Evolver](https://github.com/jMetal/Evolver) at tag `v2.4`, and are distributed under the same
license (GNU GPL v3). Evolver's training runs read them through paths relative to their working
directory (for example `resources/referenceFronts/ZDT4.csv`), so they keep Evolver's layout.

| Directory | Contents |
|---|---|
| `referenceFronts/` | Reference fronts of continuous problems, used to compute quality indicators |
| `referenceFrontsTSP/` | Reference fronts of the multi-objective TSP instances |
| `tspInstances/` | TSP instances (TSPLIB format) |
| `weightVectors/` | Weight vectors used by MOEA/D |

The reference fronts of the MaF problems (`MaF*.csv`, about 73 MB) are left out.

The copy is made by `make sync-resources` (`scripts/sync_resources.py`), which downloads the
source archive of the Evolver release set by `EVOLVER_VERSION` in `evolver_studio/evolver_client.py`
from GitHub, replaces these directories with its files and records their checksums in
`SHA256SUMS`. `tests/test_resources.py` fails if a file is missing, added or changed with respect to
`SHA256SUMS`. Run `make sync-resources` again whenever `EVOLVER_VERSION` changes.
