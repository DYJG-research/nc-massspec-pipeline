#!/usr/bin/env python3

import argparse
import json
from collections import Counter
from pathlib import Path

from multiformula_common import (
    read_jsonl,
    validate_model_row,
)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--input",
        required=True
    )

    ap.add_argument(
        "--max-formulas",
        type=int,
        default=5
    )

    ap.add_argument(
        "--stats",
        required=True
    )

    args = ap.parse_args()

    n = 0
    n_ok = 0
    n_bad = 0

    formula_count = Counter()
    polarity_count = Counter()

    bad_examples = []

    sample_ids = set()

    for line_no, row in read_jsonl(args.input):

        n += 1

        sid = row.get("sample_id")

        if sid in sample_ids:
            errors = [
                f"duplicate sample_id: {sid}"
            ]
        else:
            sample_ids.add(sid)
            errors = validate_model_row(
                row,
                max_formulas=args.max_formulas
            )

        if errors:
            n_bad += 1

            if len(bad_examples) < 30:
                bad_examples.append(
                    {
                        "line": line_no,
                        "sample_id": sid,
                        "errors": errors,
                    }
                )

        else:
            n_ok += 1

            formula_count[
                len(row["formula_hypotheses"])
            ] += 1

            polarity_count[
                str(row["polarity"]).lower()
            ] += 1

    stats = {
        "input": args.input,
        "rows": n,
        "ok": n_ok,
        "bad": n_bad,
        "formula_count_distribution":
            dict(sorted(formula_count.items())),
        "polarity_distribution":
            dict(polarity_count),
        "bad_examples": bad_examples,
    }

    Path(args.stats).parent.mkdir(
        parents=True,
        exist_ok=True
    )

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

    if n_bad:
        raise SystemExit(
            f"FAILED: {n_bad} invalid rows"
        )

    print("PASS: multiformula input is valid")


if __name__ == "__main__":
    main()
