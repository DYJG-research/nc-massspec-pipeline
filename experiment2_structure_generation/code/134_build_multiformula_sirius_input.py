#!/usr/bin/env python3

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

from multiformula_common import (
    read_jsonl,
)


def safe_filename(value):
    out = []

    for ch in value:
        if ch.isalnum() or ch in "._-":
            out.append(ch)
        else:
            out.append("_")

    return "".join(out)


def write_ms(branch, out_path):

    branch_id = branch["branch_id"]
    formula = branch["formula"]
    adduct = branch["adduct"]

    if not adduct:
        raise ValueError(
            f"{branch_id}: missing adduct"
        )

    precursor_mz = float(
        branch["precursor_mz"]
    )

    peaks = branch["peaks"]

    lines = [
        f">compound {branch_id}",
        f">feature_id {branch_id}",
        f">formula {formula}",
        f">ionization {adduct}",
        f">parentmass {precursor_mz:.10f}",
        "",
        ">ms2",
    ]

    for mz, intensity in peaks:

        mz = float(mz)
        intensity = float(intensity)

        if mz <= 0:
            continue

        if intensity < 0:
            continue

        lines.append(
            f"{mz:.10f} {intensity:.10f}"
        )

    lines.append("")

    out_path.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--branches",
        required=True
    )

    ap.add_argument(
        "--output-dir",
        required=True
    )

    ap.add_argument(
        "--stats",
        required=True
    )

    ap.add_argument(
        "--clean",
        action="store_true"
    )

    args = ap.parse_args()

    root = Path(args.output_dir)

    if args.clean and root.exists():
        shutil.rmtree(root)

    root.mkdir(
        parents=True,
        exist_ok=True
    )

    n = 0
    shard_counts = Counter()

    seen_branch_ids = set()

    for line_no, branch in read_jsonl(
        args.branches
    ):

        branch_id = branch["branch_id"]

        if branch_id in seen_branch_ids:
            raise ValueError(
                f"Duplicate branch_id: {branch_id}"
            )

        seen_branch_ids.add(branch_id)

        shard_id = int(
            branch["shard_id"]
        )

        shard_dir = (
            root /
            f"shard_{shard_id:05d}"
        )

        shard_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        filename = (
            safe_filename(branch_id) + ".ms"
        )

        write_ms(
            branch,
            shard_dir / filename
        )

        n += 1
        shard_counts[shard_id] += 1

    stats = {
        "branches":
            n,

        "num_shards":
            len(shard_counts),

        "shard_counts":
            {
                str(k): v
                for k, v
                in sorted(shard_counts.items())
            },

        "output_dir":
            str(root),
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
