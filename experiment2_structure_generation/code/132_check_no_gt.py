#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

from multiformula_common import (
    find_forbidden_paths,
)


def scan_jsonl(path):
    hits = []

    with open(path, "r", encoding="utf-8") as f:

        for line_no, line in enumerate(f, 1):

            line = line.strip()

            if not line:
                continue

            obj = json.loads(line)

            paths = find_forbidden_paths(obj)

            if paths:
                hits.append(
                    {
                        "line": line_no,
                        "paths": paths,
                    }
                )

                if len(hits) >= 100:
                    break

    return hits


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--input",
        required=True
    )

    args = ap.parse_args()

    path = Path(args.input)

    if not path.exists():
        raise SystemExit(
            f"Not found: {path}"
        )

    hits = scan_jsonl(path)

    if hits:

        print(
            json.dumps(
                hits,
                ensure_ascii=False,
                indent=2
            )
        )

        raise SystemExit(
            "FAILED: possible ground-truth leakage detected"
        )

    print(
        f"PASS: no forbidden GT fields found in {path}"
    )


if __name__ == "__main__":
    main()
