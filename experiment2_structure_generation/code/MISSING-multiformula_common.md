# ⛔ MISSING: `multiformula_common.py`

This module is **not present** in this repository. Six scripts in this directory
import it and will fail with `ModuleNotFoundError` until you provide it:

| Script | Symbols imported |
|---|---|
| `131_validate_multiformula_input.py` | validation helpers |
| `132_check_no_gt.py` | `find_forbidden_paths` |
| `133_build_multiformula_branches.py` | branch helpers |
| `134_build_multiformula_sirius_input.py` | SIRIUS-input helpers |
| `136_build_global_formula_structure_pool.py` | `read_jsonl`, `write_jsonl` |
| `140_multihyp_formula_lattice_predict.py` | `find_forbidden_paths` |

## How to obtain it

Retrieve the file from the original training server:

```
/u01/liukangjun/NC/exp3/ourllm/DistilledStructureCoT/massspec_sirius_pipeline/scripts/multiformula_common.py
```

… and drop it into **this directory** (`experiment2_structure_generation/code/`)
as a sibling of the scripts above. The six scripts do a bare
`from multiformula_common import …` (no package prefix), so a sibling file resolves
the imports with **zero edits** to those scripts.

## Why isn't it here?

The module was not included in the paper-code backup this repository was assembled
from, and the original server (`/u01/liukangjun`) is not reachable from the machine
where this release was built. It must be supplied by the authors before the
Experiment-2 pipeline can be re-run. It is **not** a data artifact — it is pure
Python helper code (JSONL I/O, forbidden-path / leakage checks, formula & adduct
normalization) owned by the project.
