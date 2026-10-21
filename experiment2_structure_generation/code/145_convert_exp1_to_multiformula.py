#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_rows(path: Path) -> list[dict[str, Any]]:
    """
    Accept either:
      1. JSON array
      2. JSONL
    """
    text = path.read_text(
        encoding="utf-8"
    ).strip()

    if not text:
        return []

    if text.startswith("["):
        data = json.loads(text)

        if not isinstance(data, list):
            raise RuntimeError(
                f"{path}: JSON root is not a list"
            )

        return data

    rows = []

    for line_no, line in enumerate(
        text.splitlines(),
        1
    ):
        if not line.strip():
            continue

        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"{path}:{line_no}: invalid JSON"
            ) from exc

        if not isinstance(row, dict):
            raise RuntimeError(
                f"{path}:{line_no}: row is not object"
            )

        rows.append(row)

    return rows


def build_dataset_index(
    rows: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:

    result = {}

    for row in rows:

        identifier = str(
            row.get("identifier")
            or row.get("id")
            or ""
        )

        if not identifier:
            continue

        if identifier in result:
            raise RuntimeError(
                f"Duplicate MassSpecGym identifier: {identifier}"
            )

        result[identifier] = row

    return result


def find_full_peaks(
    row: dict[str, Any]
) -> list[list[float]] | None:
    """
    Recover the complete MS/MS spectrum from MassSpecGym.

    Supported layouts include:

      1. [[mz, intensity], ...]
      2. [{"mz": ..., "intensity": ...}, ...]
      3. {"mzs": [...], "intensities": [...]}
      4. MassSpecGym 1.5:
         row["spectrum"]["peaks"]
    """

    def normalize_peak_list(value):
        if not isinstance(value, list):
            return None

        peaks = []

        for item in value:

            # Layout: [mz, intensity]
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                try:
                    mz = float(item[0])
                    intensity = float(item[1])
                except (TypeError, ValueError):
                    continue

            # Layout: {"mz": ..., "intensity": ...}
            elif isinstance(item, dict):
                try:
                    mz = float(item["mz"])
                    intensity = float(item["intensity"])
                except (KeyError, TypeError, ValueError):
                    continue

            else:
                continue

            if mz > 0 and intensity >= 0:
                peaks.append([mz, intensity])

        return peaks or None

    def normalize_spectrum_dict(value):
        if not isinstance(value, dict):
            return None

        # Preferred representation.
        peaks = normalize_peak_list(
            value.get("peaks")
        )

        if peaks:
            return peaks

        # Parallel arrays.
        mzs = value.get("mzs")
        intensities = value.get("intensities")

        if (
            isinstance(mzs, list)
            and isinstance(intensities, list)
            and len(mzs) == len(intensities)
        ):
            peaks = []

            for mz, intensity in zip(
                mzs,
                intensities
            ):
                try:
                    mz = float(mz)
                    intensity = float(intensity)
                except (TypeError, ValueError):
                    continue

                if mz > 0 and intensity >= 0:
                    peaks.append([mz, intensity])

            if peaks:
                return peaks

        return None

    # ----------------------------------------------------------
    # MassSpecGym 1.5 official layout:
    #
    # row["spectrum"] = {
    #     "mzs": [...],
    #     "intensities": [...],
    #     "peaks": [
    #         {"mz": ..., "intensity": ...},
    #         ...
    #     ]
    # }
    # ----------------------------------------------------------

    spectrum = row.get("spectrum")

    peaks = normalize_spectrum_dict(
        spectrum
    )

    if peaks:
        return peaks

    # Other possible nested layouts.
    for key in (
        "ms2",
        "ms2_spectrum",
    ):
        peaks = normalize_spectrum_dict(
            row.get(key)
        )

        if peaks:
            return peaks

    # Possible top-level peak arrays.
    for key in (
        "peaks",
        "ms2_peaks",
    ):
        peaks = normalize_peak_list(
            row.get(key)
        )

        if peaks:
            return peaks

    return None

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--exp1",
        required=True,
        help="Experiment-1 JSON/JSONL output"
    )

    ap.add_argument(
        "--massspecgym",
        required=True,
        help="Original MassSpecGym JSON/JSONL"
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
        "--top-k",
        type=int,
        default=5
    )

    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="0 means all"
    )

    ap.add_argument(
        "--allow-prompt-peaks",
        action="store_true",
        help=(
            "Engineering fallback only. "
            "Use prompt_peaks when original full spectrum "
            "cannot be found."
        )
    )

    args = ap.parse_args()

    exp1_rows = load_rows(
        Path(args.exp1)
    )

    dataset_rows = load_rows(
        Path(args.massspecgym)
    )

    dataset_by_id = build_dataset_index(
        dataset_rows
    )

    if args.limit > 0:
        exp1_rows = exp1_rows[
            :args.limit
        ]

    output_rows = []

    missing_dataset = []
    missing_peaks = []
    malformed_shortlist = []
    shortlist_shorter_than_k = 0
    score_length_mismatch = 0

    formula_counts = {}
    element_tokens = {}

    for row in exp1_rows:

        identifier = str(
            row.get("identifier")
            or ""
        )

        if not identifier:
            raise RuntimeError(
                "Experiment-1 row has no identifier"
            )

        original = dataset_by_id.get(
            identifier
        )

        if original is None:
            missing_dataset.append(
                identifier
            )
            continue

        peaks = find_full_peaks(
            original
        )

        if (
            not peaks
            and args.allow_prompt_peaks
        ):
            prompt = row.get(
                "prompt_peaks"
            )

            if isinstance(prompt, list):
                peaks = []

                for pair in prompt:
                    try:
                        mz = float(pair[0])
                        intensity = float(
                            pair[1]
                        )
                    except Exception:
                        continue

                    if (
                        mz > 0
                        and intensity >= 0
                    ):
                        peaks.append(
                            [mz, intensity]
                        )

        if not peaks:
            missing_peaks.append(
                identifier
            )
            continue

        shortlist = row.get(
            "shortlist"
        )

        scores = row.get(
            "preranker_scores"
        )

        if (
            not isinstance(shortlist, list)
            or not shortlist
        ):
            malformed_shortlist.append(
                identifier
            )
            continue

        if not isinstance(scores, list):
            scores = []

        if (
            scores
            and len(scores) != len(shortlist)
        ):
            score_length_mismatch += 1

        selected = shortlist[
            :args.top_k
        ]

        if len(selected) < args.top_k:
            shortlist_shorter_than_k += 1

        hypotheses = []

        seen_formulas = set()

        for idx, candidate in enumerate(
            selected,
            1
        ):

            if not isinstance(
                candidate,
                dict
            ):
                continue

            formula = str(
                candidate.get(
                    "formula"
                )
                or ""
            ).strip()

            adduct = str(
                candidate.get(
                    "best_adduct"
                )
                or ""
            ).strip()

            if (
                not formula
                or not adduct
            ):
                continue

            # Do not duplicate the same neutral formula.
            if formula in seen_formulas:
                continue

            seen_formulas.add(
                formula
            )

            raw_score = (
                scores[idx - 1]
                if idx - 1 < len(scores)
                else None
            )

            try:
                raw_score = (
                    float(raw_score)
                    if raw_score is not None
                    else None
                )
            except (
                TypeError,
                ValueError
            ):
                raw_score = None

            hypothesis = {
                "rank":
                    len(hypotheses) + 1,

                "formula":
                    formula,

                "adduct":
                    adduct,

                # Raw Experiment-1 preranker score.
                "score":
                    raw_score,

                # Do NOT invent a probability.
                "confidence":
                    None,

                "ppm_error":
                    candidate.get(
                        "ppm_error"
                    ),

                "dbe":
                    candidate.get(
                        "dbe"
                    ),

                # Optional provenance fields.
                "support":
                    candidate.get(
                        "support"
                    ),

                "v3_rank":
                    candidate.get(
                        "v3_rank"
                    ),

                "old_rank":
                    candidate.get(
                        "old_rank"
                    ),

                "direct_rank":
                    candidate.get(
                        "direct_rank"
                    ),

                "f_rank":
                    candidate.get(
                        "f_rank"
                    ),
            }

            hypotheses.append(
                hypothesis
            )

            formula_counts[
                formula
            ] = formula_counts.get(
                formula,
                0
            ) + 1

        if not hypotheses:
            malformed_shortlist.append(
                identifier
            )
            continue

        precursor_mz = row.get(
            "precursor_mz"
        )

        if precursor_mz is None:
            precursor_mz = original.get(
                "precursor_mz"
            )

        polarity = (
            row.get("polarity")
            or original.get(
                "polarity"
            )
        )

        output_rows.append(
            {
                "sample_id":
                    identifier,

                "precursor_mz":
                    float(
                        precursor_mz
                    ),

                "polarity":
                    polarity,

                "peaks":
                    peaks,

                "formula_hypotheses":
                    hypotheses,
            }
        )

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with output_path.open(
        "w",
        encoding="utf-8"
    ) as f:

        for row in output_rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False
                )
                + "\n"
            )

    stats = {
        "exp1_rows_considered":
            len(exp1_rows),

        "output_rows":
            len(output_rows),

        "top_k":
            args.top_k,

        "missing_massspecgym_rows":
            len(missing_dataset),

        "missing_full_spectrum_rows":
            len(missing_peaks),

        "malformed_shortlist_rows":
            len(malformed_shortlist),

        "shortlist_shorter_than_k":
            shortlist_shorter_than_k,

        "preranker_score_length_mismatch":
            score_length_mismatch,

        "unique_predicted_formulas":
            len(formula_counts),

        "examples_missing_massspecgym":
            missing_dataset[:10],

        "examples_missing_peaks":
            missing_peaks[:10],

        "examples_malformed_shortlist":
            malformed_shortlist[:10],

        "truth_fields_written":
            [],

        "prompt_peaks_used":
            bool(
                args.allow_prompt_peaks
            ),
    }

    stats_path = Path(
        args.stats
    )

    stats_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    stats_path.write_text(
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

    print(
        "[OK] Multiformula input:",
        output_path
    )


if __name__ == "__main__":
    main()
