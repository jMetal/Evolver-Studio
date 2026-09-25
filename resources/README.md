# Resources copied from Evolver

The files in this directory are copied, unchanged, from the `resources/` directory of
[Evolver](https://github.com/jMetal/Evolver) at tag `v2.1`, and are distributed under the same
license (GNU GPL v3). Evolver's training runs read them through paths relative to their working
directory (for example `resources/referenceFronts/ZDT4.csv`), so they keep Evolver's layout.

| Directory | Contents |
|---|---|
| `referenceFronts/` | Reference fronts of continuous problems, used to compute quality indicators |
| `referenceFrontsTSP/` | Reference fronts of the multi-objective TSP instances |
| `tspInstances/` | TSP instances (TSPLIB format) |
| `weightVectors/` | Weight vectors used by MOEA/D |

The reference fronts of the MaF problems (`MaF*.csv`, about 73 MB) are left out.

`tests/test_resources.py` checks that these files match those of an Evolver checkout: it fails if
Evolver adds, removes or changes a file. To refresh the copy from a new Evolver release:

```bash
git -C <evolver-checkout> archive <tag> resources/referenceFronts resources/referenceFrontsTSP \
    resources/tspInstances resources/weightVectors | tar -x -C .
rm resources/referenceFronts/MaF*
```
