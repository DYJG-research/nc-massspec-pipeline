#!/usr/bin/env bash
set -Eeuo pipefail


PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT is required}"

BRANCH_MANIFEST="${BRANCH_MANIFEST:?BRANCH_MANIFEST is required}"
MERGED_CANDIDATES="${MERGED_CANDIDATES:?MERGED_CANDIDATES is required}"
LATTICE_ROOT="${LATTICE_ROOT:?LATTICE_ROOT is required}"


LATTICE_TOPK="${LATTICE_TOPK:-10}"
LATTICE_LOCK_MODE="${LATTICE_LOCK_MODE:-top1}"
PROGRESS_EVERY="${PROGRESS_EVERY:-100}"


cd "$PROJECT_ROOT"


mkdir -p \
  "$LATTICE_ROOT"


LATTICE_CANDIDATES="$LATTICE_ROOT/branch_candidates_after_lattice.jsonl"

LATTICE_AUDIT="$LATTICE_ROOT/lattice_audit.json"

GLOBAL_POOL="$LATTICE_ROOT/global_formula_structure_pool.jsonl"

GLOBAL_STATS="$LATTICE_ROOT/global_pool_stats.json"


expected_branches="$(
  wc -l < "$BRANCH_MANIFEST"
)"


expected_samples="$(
python - "$BRANCH_MANIFEST" <<'PY'
import json
import sys

p = sys.argv[1]

ids = set()

with open(
    p,
    "r",
    encoding="utf-8"
) as f:

    for line in f:

        if not line.strip():
            continue

        row = json.loads(line)

        ids.add(
            str(row["sample_id"])
        )

print(len(ids))
PY
)"


candidate_rows="$(
  wc -l < "$MERGED_CANDIDATES"
)"


if [[ "$candidate_rows" -ne "$expected_branches" ]]; then

    echo \
      "[ERROR] Candidate/branch mismatch before lattice: " \
      "branches=$expected_branches " \
      "candidates=$candidate_rows" \
      >&2

    exit 1
fi


echo "============================================================"
echo "GLOBAL MULTI-HYPOTHESIS FORMULA-LATTICE"
echo "============================================================"
echo "Samples:          $expected_samples"
echo "Formula branches: $expected_branches"
echo "Top-K/branch:     $LATTICE_TOPK"
echo "Lock mode:        $LATTICE_LOCK_MODE"
echo "============================================================"


python -u \
  scripts/140_multihyp_formula_lattice_predict.py \
  --branches "$BRANCH_MANIFEST" \
  --branch-candidates "$MERGED_CANDIDATES" \
  --output "$LATTICE_CANDIDATES" \
  --audit "$LATTICE_AUDIT" \
  --expected-branches "$expected_branches" \
  --expected-samples "$expected_samples" \
  --top-k "$LATTICE_TOPK" \
  --lock-mode "$LATTICE_LOCK_MODE" \
  --progress-every "$PROGRESS_EVERY"


echo
echo "============================================================"
echo "BUILDING GLOBAL FORMULA-STRUCTURE POOL"
echo "============================================================"


python -u \
  scripts/136_build_global_formula_structure_pool.py \
  --branch-candidates "$LATTICE_CANDIDATES" \
  --output "$GLOBAL_POOL" \
  --stats "$GLOBAL_STATS"


echo
echo "============================================================"
echo "VERIFYING WITHIN-BRANCH TOP-1 PRESERVATION"
echo "============================================================"


export BRANCH_MANIFEST
export MERGED_CANDIDATES
export LATTICE_CANDIDATES


python - <<'PY'
import json
import os
from pathlib import Path


def read_rows(path):

    return [
        json.loads(line)
        for line in Path(path).open(
            "r",
            encoding="utf-8"
        )
        if line.strip()
    ]


before = {
    str(row["branch_id"]): row
    for row in read_rows(
        os.environ[
            "MERGED_CANDIDATES"
        ]
    )
}


after = {
    str(row["branch_id"]): row
    for row in read_rows(
        os.environ[
            "LATTICE_CANDIDATES"
        ]
    )
}


if set(before) != set(after):

    raise RuntimeError(
        "Branch set changed during Formula-Lattice"
    )


checked = 0


for branch_id, row_before in before.items():

    c_before = (
        row_before.get("candidates")
        or []
    )

    if not c_before:
        continue

    c_after = (
        after[
            branch_id
        ].get("candidates")
        or []
    )

    if not c_after:

        raise RuntimeError(
            f"Originally nonempty branch became empty: "
            f"{branch_id}"
        )

    expected = c_before[0]["smiles"]

    actual = c_after[0]["smiles"]

    if expected != actual:

        raise RuntimeError(
            f"Top-1 lock violation: "
            f"{branch_id}: "
            f"before={expected} "
            f"after={actual}"
        )

    checked += 1


print(
    f"PASS: Top-1 preserved for "
    f"{checked} originally nonempty branches"
)
PY


echo
echo "============================================================"
echo "[OK] GLOBAL FORMULA-LATTICE COMPLETE"
echo "============================================================"
echo "Lattice candidates: $LATTICE_CANDIDATES"
echo "Lattice audit:      $LATTICE_AUDIT"
echo "Global pool:        $GLOBAL_POOL"
echo "Global stats:       $GLOBAL_STATS"
echo "============================================================"
