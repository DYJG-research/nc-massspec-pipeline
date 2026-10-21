#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise RuntimeError(f"{path}:{line_number}: row is not an object")
            rows.append(row)
    return rows


def safe_filename(identifier: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", identifier)
    if not name:
        raise RuntimeError(f"Cannot create filename for identifier={identifier!r}")
    return name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create SIRIUS .ms files from the fixed test1000 selected JSONL."
    )
    parser.add_argument("--selected", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(Path(args.selected))
    if len(rows) != args.expected_rows:
        raise RuntimeError(
            f"Expected {args.expected_rows} rows, found {len(rows)} in {args.selected}"
        )

    output_dir = Path(args.output_dir)
    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    required = {"identifier", "true_formula", "precursor_mz", "adduct", "peaks"}
    seen_ids = set()
    written = 0

    for row in rows:
        missing = required - set(row)
        if missing:
            raise RuntimeError(
                f"{row.get('identifier', '<unknown>')}: missing {sorted(missing)}"
            )

        identifier = str(row["identifier"])
        if identifier in seen_ids:
            raise RuntimeError(f"Duplicate identifier: {identifier}")
        seen_ids.add(identifier)

        peaks = []
        for pair in row.get("peaks") or []:
            if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                continue
            try:
                mz = float(pair[0])
                intensity = float(pair[1])
            except (TypeError, ValueError):
                continue
            if mz > 0 and intensity >= 0:
                peaks.append((mz, intensity))

        if not peaks:
            raise RuntimeError(f"{identifier}: no valid MS2 peaks")

        path = output_dir / f"{safe_filename(identifier)}.ms"
        with path.open("w", encoding="utf-8") as handle:
            handle.write(f">compound {identifier}\n")
            handle.write(f">parentmass {float(row['precursor_mz']):.10f}\n")
            handle.write(f">ionization {row['adduct']}\n")
            handle.write(f">formula {row['true_formula']}\n")
            handle.write(">ms2\n")
            for mz, intensity in peaks:
                handle.write(f"{mz:.10f} {intensity:.10f}\n")

        written += 1
        if written % 100 == 0 or written == len(rows):
            print(f"[MS] {written}/{len(rows)}", flush=True)

    print("[OK] Wrote .ms files:", written)
    print("[OK] Output directory:", output_dir)


if __name__ == "__main__":
    main()
