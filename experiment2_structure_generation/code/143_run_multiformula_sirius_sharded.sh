#!/usr/bin/env bash
set -Eeuo pipefail


PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT is required}"
SIRIUS_BIN="${SIRIUS_BIN:?SIRIUS_BIN is required}"

BRANCH_MANIFEST="${BRANCH_MANIFEST:?BRANCH_MANIFEST is required}"
SIRIUS_INPUT_ROOT="${SIRIUS_INPUT_ROOT:?SIRIUS_INPUT_ROOT is required}"
RUN_ROOT="${RUN_ROOT:?RUN_ROOT is required}"
MERGED_CANDIDATES="${MERGED_CANDIDATES:?MERGED_CANDIDATES is required}"


SIRIUS_CPUSET="${SIRIUS_CPUSET:-0-31}"
SIRIUS_TOPK="${SIRIUS_TOPK:-100}"
SIRIUS_PROFILE="${SIRIUS_PROFILE:-orbitrap}"
SIRIUS_PPM="${SIRIUS_PPM:-10}"

REBUILD_SHARDS="${REBUILD_SHARDS:-0}"


cd "$PROJECT_ROOT"


mkdir -p \
  "$RUN_ROOT" \
  "$RUN_ROOT/logs" \
  "$(dirname "$MERGED_CANDIDATES")"


SHARD_MANIFEST_DIR="$RUN_ROOT/branch_shards"

SHARD_STATS="$RUN_ROOT/branch_shards_stats.json"


echo "============================================================"
echo "BUILDING BRANCH SHARD MANIFESTS"
echo "============================================================"


python -u \
  scripts/142_build_branch_shard_manifests.py \
  --branches "$BRANCH_MANIFEST" \
  --output-dir "$SHARD_MANIFEST_DIR" \
  --stats "$SHARD_STATS" \
  --clean


shard_files=(
  "$SHARD_MANIFEST_DIR"/shard_*.jsonl
)


if [[ ! -e "${shard_files[0]}" ]]; then

    echo \
      "[ERROR] No shard manifests found." \
      >&2

    exit 1
fi


echo
echo "[INFO] Number of shards: ${#shard_files[@]}"


for shard_manifest in "${shard_files[@]}"; do

    tag="$(
      basename "$shard_manifest" .jsonl
    )"

    expected="$(
      wc -l < "$shard_manifest"
    )"

    input_dir="$SIRIUS_INPUT_ROOT/$tag"

    out_dir="$RUN_ROOT/$tag"

    project="$out_dir/${tag}.sirius"

    summary_dir="$out_dir/summaries"

    candidates="$out_dir/branch_candidates.jsonl"

    audit="$out_dir/parse_audit.json"


    echo
    echo "============================================================"
    echo "SHARD:      $tag"
    echo "Branches:   $expected"
    echo "Input:      $input_dir"
    echo "Output:     $out_dir"
    echo "============================================================"


    if [[ ! -d "$input_dir" ]]; then

        echo \
          "[ERROR] Missing SIRIUS input directory: $input_dir" \
          >&2

        exit 1
    fi


    ms_count="$(
      find "$input_dir" \
        -maxdepth 1 \
        -type f \
        -name "*.ms" \
        | wc -l
    )"


    if [[ "$ms_count" -ne "$expected" ]]; then

        echo \
          "[ERROR] $tag branch/.ms mismatch: " \
          "branches=$expected ms=$ms_count" \
          >&2

        exit 1
    fi


    if [[ "$REBUILD_SHARDS" != "1" ]] \
       && [[ -s "$candidates" ]] \
       && [[ "$(wc -l < "$candidates")" -eq "$expected" ]]; then

        echo \
          "[SKIP] $tag already complete: $expected branches"

        continue
    fi


    rm -rf "$out_dir"

    mkdir -p \
      "$out_dir/logs" \
      "$summary_dir"


    prefix=()

    if [[ -n "$SIRIUS_CPUSET" ]]; then

        prefix=(
          taskset -c "$SIRIUS_CPUSET"
        )

    fi


    echo
    echo "[SIRIUS] $tag"
    echo "[SIRIUS] profile=$SIRIUS_PROFILE"
    echo "[SIRIUS] ppm=$SIRIUS_PPM"
    echo "[SIRIUS] candidates=$SIRIUS_TOPK"


    set +e


    "${prefix[@]}" "$SIRIUS_BIN" \
      --input "$input_dir" \
      --project "$project" \
      formulas \
      -p "$SIRIUS_PROFILE" \
      --ppm-max "$SIRIUS_PPM" \
      --ppm-max-ms2 "$SIRIUS_PPM" \
      fingerprints \
      classes \
      denovo-structures \
      --candidates "$SIRIUS_TOPK" \
      2>&1 \
      | tee "$out_dir/logs/sirius_compute.log"


    sirius_status=${PIPESTATUS[0]}


    set -e


    if [[ "$sirius_status" -ne 0 ]]; then

        if grep -q \
          "Workflow has been finished" \
          "$out_dir/logs/sirius_compute.log"; then

            echo \
              "[WARN] SIRIUS returned $sirius_status " \
              "after workflow completion."

        else

            echo \
              "[ERROR] SIRIUS failed for $tag, " \
              "status=$sirius_status" \
              >&2

            exit "$sirius_status"

        fi
    fi


    echo
    echo "[EXPORT] $tag"


    set +e


    "$SIRIUS_BIN" \
      --project "$project" \
      write-summaries \
      --output "$summary_dir" \
      --format tsv \
      --top-k-summary "$SIRIUS_TOPK" \
      2>&1 \
      | tee "$out_dir/logs/sirius_export.log"


    export_status=${PIPESTATUS[0]}


    set -e


    summary="$(
      find "$summary_dir" \
        -maxdepth 1 \
        -type f \
        -name 'denovo_structure_identifications_top-*.tsv' \
        | sort \
        | tail -1
    )"


    # A shard may legitimately produce zero MSNovelist structures.
    #
    # In that case SIRIUS may not create a denovo summary at all.
    # Create a header-only summary so the parser emits explicit
    # empty candidate rows rather than silently losing branches.
    if [[ -z "$summary" || ! -s "$summary" ]]; then

        echo \
          "[WARN] $tag produced no de novo summary; " \
          "creating an explicit empty summary."

        summary="$summary_dir/denovo_structure_identifications_top-${SIRIUS_TOPK}.tsv"

        printf \
          'structurePerIdRank\tmolecularFormula\tsmiles\tmappingFeatureId\n' \
          > "$summary"

        echo \
          "[WARN] SIRIUS export status=$export_status"

    fi


    echo
    echo "[PARSE] $tag"
    echo "[PARSE] Summary: $summary"


    python -u \
      scripts/138_parse_multiformula_denovo_summary.py \
      --branches "$shard_manifest" \
      --summary "$summary" \
      --output "$candidates" \
      --audit "$audit" \
      --expected-branches "$expected" \
      --top-k "$SIRIUS_TOPK" \
      2>&1 \
      | tee "$out_dir/logs/parse.log"


    parsed="$(
      wc -l < "$candidates"
    )"


    if [[ "$parsed" -ne "$expected" ]]; then

        echo \
          "[ERROR] $tag parsed row mismatch: " \
          "expected=$expected actual=$parsed" \
          >&2

        exit 1
    fi


    echo \
      "[OK] $tag complete: $parsed branches"

done


echo
echo "============================================================"
echo "MERGING ALL FORMULA BRANCHES"
echo "============================================================"


export BRANCH_MANIFEST
export RUN_ROOT
export MERGED_CANDIDATES


python - <<'PY'
import json
import os
from pathlib import Path


manifest_path = Path(
    os.environ["BRANCH_MANIFEST"]
)

run_root = Path(
    os.environ["RUN_ROOT"]
)

output_path = Path(
    os.environ["MERGED_CANDIDATES"]
)


manifest_rows = [
    json.loads(line)
    for line in manifest_path.open(
        "r",
        encoding="utf-8"
    )
    if line.strip()
]


branch_order = [
    str(row["branch_id"])
    for row in manifest_rows
]


if len(branch_order) != len(set(branch_order)):
    raise RuntimeError(
        "Duplicate branch IDs in global manifest"
    )


candidate_by_branch = {}


source_files = sorted(
    run_root.glob(
        "shard_*/branch_candidates.jsonl"
    )
)


if not source_files:
    raise RuntimeError(
        "No shard candidate files found"
    )


print(
    "Shard candidate files:",
    len(source_files)
)


for path in source_files:

    count = 0

    with path.open(
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            if not line.strip():
                continue

            row = json.loads(line)

            branch_id = str(
                row["branch_id"]
            )

            if branch_id in candidate_by_branch:

                raise RuntimeError(
                    f"Duplicate branch candidate row: "
                    f"{branch_id}"
                )

            candidate_by_branch[
                branch_id
            ] = row

            count += 1

    print(
        path.parent.name,
        count
    )


manifest_set = set(
    branch_order
)

candidate_set = set(
    candidate_by_branch
)


missing = sorted(
    manifest_set
    - candidate_set
)

extra = sorted(
    candidate_set
    - manifest_set
)


if missing:

    raise RuntimeError(
        f"Missing {len(missing)} branches; "
        f"examples={missing[:20]}"
    )


if extra:

    raise RuntimeError(
        f"Unexpected {len(extra)} branches; "
        f"examples={extra[:20]}"
    )


output_path.parent.mkdir(
    parents=True,
    exist_ok=True
)


with output_path.open(
    "w",
    encoding="utf-8"
) as f:

    for branch_id in branch_order:

        f.write(
            json.dumps(
                candidate_by_branch[
                    branch_id
                ],
                ensure_ascii=False
            )
            + "\n"
        )


nonempty = sum(
    bool(
        candidate_by_branch[
            branch_id
        ].get("candidates")
    )
    for branch_id in branch_order
)


candidate_total = sum(
    len(
        candidate_by_branch[
            branch_id
        ].get(
            "candidates",
            []
        )
    )
    for branch_id in branch_order
)


audit = {
    "branches":
        len(branch_order),

    "shard_candidate_files":
        len(source_files),

    "nonempty_branches":
        nonempty,

    "empty_branches":
        len(branch_order)
        - nonempty,

    "branch_coverage":
        (
            nonempty
            / len(branch_order)
            if branch_order
            else 0.0
        ),

    "candidate_total":
        candidate_total,

    "average_candidates_per_branch":
        (
            candidate_total
            / len(branch_order)
            if branch_order
            else 0.0
        ),

    "output":
        str(output_path),
}


audit_path = (
    output_path.parent /
    (
        output_path.stem
        + "_merge_audit.json"
    )
)


audit_path.write_text(
    json.dumps(
        audit,
        ensure_ascii=False,
        indent=2
    ),
    encoding="utf-8"
)


print(
    json.dumps(
        audit,
        ensure_ascii=False,
        indent=2
    )
)
PY


merged_rows="$(
  wc -l < "$MERGED_CANDIDATES"
)"


expected_total="$(
  wc -l < "$BRANCH_MANIFEST"
)"


if [[ "$merged_rows" -ne "$expected_total" ]]; then

    echo \
      "[ERROR] Final merge mismatch: " \
      "expected=$expected_total actual=$merged_rows" \
      >&2

    exit 1
fi


echo
echo "============================================================"
echo "[OK] ALL SIRIUS SHARDS COMPLETE"
echo "============================================================"
echo "Branches:   $merged_rows"
echo "Output:     $MERGED_CANDIDATES"
echo "============================================================"
