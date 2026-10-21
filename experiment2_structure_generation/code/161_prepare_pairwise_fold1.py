#!/usr/bin/env python3

import argparse
import json
import random
from collections import Counter
from pathlib import Path


def read_jsonl(path):
    rows = []

    with open(
        path,
        encoding="utf-8",
    ) as f:
        for line in f:
            if line.strip():
                rows.append(
                    json.loads(line)
                )

    return rows


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--input",
        required=True,
    )

    ap.add_argument(
        "--target-fold",
        type=int,
        default=1,
    )

    ap.add_argument(
        "--output-dir",
        required=True,
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=20260820,
    )

    args = ap.parse_args()

    rows = read_jsonl(
        args.input
    )

    train_full = [
        row
        for row in rows
        if int(row["fold"])
        != args.target_fold
    ]

    holdout = [
        row
        for row in rows
        if int(row["fold"])
        == args.target_fold
    ]

    train_ids = {
        x["sample_id"]
        for x in train_full
    }

    holdout_ids = {
        x["sample_id"]
        for x in holdout
    }

    overlap = (
        train_ids
        & holdout_ids
    )

    if overlap:
        raise RuntimeError(
            f"Sample leakage: "
            f"{len(overlap)}"
        )

    # Only messages are visible to ms-swift.
    train = [
        {
            "messages":
                row["messages"]
        }
        for row in train_full
    ]

    rng = random.Random(
        args.seed
    )

    rng.shuffle(train)

    outdir = Path(
        args.output_dir
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with (
        outdir / "train.jsonl"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        for row in train:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    pair_types = Counter(
        x["pair_type"]
        for x in train_full
    )

    targets = Counter(
        x["messages"][-1]["content"]
        for x in train_full
    )

    stats = {
        "target_fold":
            args.target_fold,

        "train_pair_rows":
            len(train),

        "train_unique_samples":
            len(train_ids),

        "holdout_pair_rows":
            len(holdout),

        "holdout_unique_samples":
            len(holdout_ids),

        "train_holdout_overlap":
            len(overlap),

        "train_pair_types":
            dict(pair_types),

        "train_target_distribution":
            dict(targets),
    }

    (
        outdir
        / "split_stats.json"
    ).write_text(
        json.dumps(
            stats,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            stats,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
