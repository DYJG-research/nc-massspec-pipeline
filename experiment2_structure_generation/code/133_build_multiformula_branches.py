#!/usr/bin/env python3

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from multiformula_common import (
    read_jsonl,
    validate_model_row,
    canonical_formula,
    normalize_polarity,
    normalize_adduct,
    write_jsonl,
)


def make_branch_id(sample_id, rank, formula, adduct):

    raw = (
        f"{sample_id}|{rank}|{formula}|{adduct}"
    )

    short_hash = hashlib.sha1(
        raw.encode("utf-8")
    ).hexdigest()[:10]

    return (
        f"{sample_id}__F{rank:02d}__{short_hash}"
    )


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--input",
        required=True
    )

    ap.add_argument(
        "--output",
        required=True
    )

    ap.add_argument(
        "--stats",
        required=True
    )

    ap.add_argument(
        "--max-formulas",
        type=int,
        default=5
    )

    ap.add_argument(
        "--shard-size",
        type=int,
        default=500
    )

    args = ap.parse_args()

    branches = []

    sample_count = 0
    branch_count = 0

    per_rank = Counter()

    for line_no, row in read_jsonl(args.input):

        errors = validate_model_row(
            row,
            max_formulas=args.max_formulas
        )

        if errors:
            raise ValueError(
                f"Invalid row {line_no}: {errors}"
            )

        sample_count += 1

        sid = row["sample_id"]

        precursor_mz = float(
            row["precursor_mz"]
        )

        polarity = normalize_polarity(
            row["polarity"]
        )

        peaks = [
            [float(x[0]), float(x[1])]
            for x in row["peaks"]
        ]

        hypotheses = sorted(
            row["formula_hypotheses"],
            key=lambda x: int(x["rank"])
        )

        for h in hypotheses:

            rank = int(h["rank"])
            formula = canonical_formula(
                h["formula"]
            )
            adduct = normalize_adduct(
                h["adduct"]
            )

            branch_id = make_branch_id(
                sid,
                rank,
                formula,
                adduct
            )

            shard_id = (
                branch_count // args.shard_size
            )

            branch = {
                "schema_version":
                    "multiformula-branch-v1",

                "sample_id":
                    sid,

                "branch_id":
                    branch_id,

                "shard_id":
                    shard_id,

                "formula_rank":
                    rank,

                "formula":
                    formula,

                "adduct":
                    adduct,

                "formula_score":
                    (
                        float(h["score"])
                        if h.get("score") is not None
                        else None
                    ),

                "formula_confidence":
                    (
                        float(h["confidence"])
                        if h.get("confidence") is not None
                        else None
                    ),

                "formula_ppm_error":
                    (
                        float(h["ppm_error"])
                        if h.get("ppm_error") is not None
                        else None
                    ),

                "formula_dbe":
                    (
                        float(h["dbe"])
                        if h.get("dbe") is not None
                        else None
                    ),

                "precursor_mz":
                    precursor_mz,

                "polarity":
                    polarity,

                "peaks":
                    peaks,
            }

            branches.append(branch)

            branch_count += 1
            per_rank[rank] += 1

    write_jsonl(
        args.output,
        branches
    )

    stats = {
        "input":
            args.input,

        "samples":
            sample_count,

        "branches":
            branch_count,

        "max_formulas":
            args.max_formulas,

        "shard_size":
            args.shard_size,

        "num_shards":
            (
                (branch_count + args.shard_size - 1)
                // args.shard_size
            ),

        "branches_by_formula_rank":
            dict(sorted(per_rank.items())),
    }

    Path(args.stats).write_text(
        json.dumps(
            stats,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print(
        json.dumps(
            stats,
            ensure_ascii=False,
            indent=2
        )
    )


if __name__ == "__main__":
    main()
