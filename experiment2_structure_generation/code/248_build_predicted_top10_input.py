#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def load_chemistry(path: Path):

    spec = importlib.util.spec_from_file_location(
        "top10_formula_chemistry",
        str(path),
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Cannot load chemistry module: {path}"
        )

    mod = importlib.util.module_from_spec(spec)

    # Required for dataclass under Python 3.11
    sys.modules[spec.name] = mod

    spec.loader.exec_module(mod)

    return mod


def read_jsonl(path: Path):

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line_no, line in enumerate(f, 1):

            line = line.strip()

            if not line:
                continue

            try:
                yield json.loads(line)

            except Exception as exc:

                raise RuntimeError(
                    f"{path}:{line_no}: {exc}"
                ) from exc


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--exp1",
        required=True,
    )

    ap.add_argument(
        "--top5-branches",
        required=True,
    )

    ap.add_argument(
        "--chemistry",
        required=True,
    )

    ap.add_argument(
        "--output",
        required=True,
    )

    ap.add_argument(
        "--stats",
        required=True,
    )

    args = ap.parse_args()

    exp1_path = Path(args.exp1)
    top5_path = Path(args.top5_branches)
    chemistry_path = Path(args.chemistry)

    output_path = Path(args.output)
    stats_path = Path(args.stats)

    chemistry = load_chemistry(
        chemistry_path
    )

    normalize_formula = (
        chemistry.normalize_formula
    )

    # ========================================================
    # Load Exp1 in original order
    # ========================================================

    exp1 = {}
    sample_order = []

    for row in read_jsonl(exp1_path):

        sid = str(
            row["identifier"]
        )

        if sid in exp1:
            raise RuntimeError(
                f"Duplicate Exp1 identifier: {sid}"
            )

        exp1[sid] = row
        sample_order.append(sid)

    if len(sample_order) != 17556:
        raise RuntimeError(
            f"Exp1 samples={len(sample_order)}, "
            f"expected 17556"
        )

    # ========================================================
    # Load existing Top-5 branches
    # ========================================================

    top5_by_sample = defaultdict(list)

    for row in read_jsonl(top5_path):

        sid = str(
            row["sample_id"]
        )

        top5_by_sample[sid].append(
            row
        )

    if len(top5_by_sample) != 17556:
        raise RuntimeError(
            "Existing Top-5 does not contain "
            "17556 unique samples"
        )

    for sid in top5_by_sample:

        top5_by_sample[sid].sort(
            key=lambda x: int(
                x["formula_rank"]
            )
        )

    # ========================================================
    # Construct Top-10 input
    # ========================================================

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rank_counts = Counter()

    total_hypotheses = 0

    samples_with_lt10 = 0

    rank15_formula_check_fail = 0

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as fout:

        for sid in sample_order:

            exp = exp1[sid]

            old_rows = top5_by_sample[
                sid
            ]

            if not old_rows:
                raise RuntimeError(
                    f"No existing Top-5 branch: {sid}"
                )

            # ------------------------------------------------
            # Observable information MUST come from
            # existing Top-5 branch, not reconstructed
            # from Exp1.
            # ------------------------------------------------

            first = old_rows[0]

            precursor_mz = float(
                first["precursor_mz"]
            )

            polarity = first["polarity"]

            peaks = first.get(
                "peaks"
            ) or []

            shortlist = exp.get(
                "shortlist"
            ) or []

            scores = exp.get(
                "preranker_scores"
            ) or []

            if len(shortlist) < 10:
                samples_with_lt10 += 1

            max_rank = min(
                10,
                len(shortlist),
                len(scores),
            )

            hypotheses = []

            # =================================================
            # Rank 1-5:
            # EXACTLY inherit existing Top-5 branch metadata.
            # =================================================

            for old in old_rows:

                rank = int(
                    old["formula_rank"]
                )

                if rank > 5:
                    continue

                # Extra consistency check against Exp1
                # after formula normalization.
                if rank <= len(shortlist):

                    a = normalize_formula(
                        str(
                            old["formula"]
                        )
                    )

                    b = normalize_formula(
                        str(
                            shortlist[
                                rank - 1
                            ]["formula"]
                        )
                    )

                    if a != b:
                        rank15_formula_check_fail += 1

                hypotheses.append({
                    "rank":
                        rank,

                    "formula":
                        old["formula"],

                    "adduct":
                        old["adduct"],

                    "score":
                        old.get(
                            "formula_score"
                        ),

                    "confidence":
                        old.get(
                            "formula_confidence"
                        ),

                    "ppm_error":
                        old.get(
                            "formula_ppm_error"
                        ),

                    "dbe":
                        old.get(
                            "formula_dbe"
                        ),
                })

                rank_counts[
                    rank
                ] += 1

                total_hypotheses += 1

            # =================================================
            # Rank 6-10:
            # Add from Exp1 shortlist.
            # =================================================

            for rank in range(
                6,
                max_rank + 1,
            ):

                item = shortlist[
                    rank - 1
                ]

                raw_formula = str(
                    item["formula"]
                )

                formula = normalize_formula(
                    raw_formula
                )

                if not formula:
                    raise RuntimeError(
                        f"{sid}: cannot normalize "
                        f"formula rank {rank}: "
                        f"{raw_formula}"
                    )

                adduct = item.get(
                    "best_adduct"
                )

                if not adduct:
                    raise RuntimeError(
                        f"{sid}: missing best_adduct "
                        f"at rank {rank}"
                    )

                hypotheses.append({
                    "rank":
                        rank,

                    "formula":
                        formula,

                    "adduct":
                        adduct,

                    "score":
                        float(
                            scores[
                                rank - 1
                            ]
                        ),

                    # Current Top-5 has confidence=None
                    # for every branch, therefore Top-10
                    # must retain the same definition.
                    "confidence":
                        None,

                    "ppm_error":
                        item.get(
                            "ppm_error"
                        ),

                    "dbe":
                        item.get(
                            "dbe"
                        ),
                })

                rank_counts[
                    rank
                ] += 1

                total_hypotheses += 1

            hypotheses.sort(
                key=lambda x: int(
                    x["rank"]
                )
            )

            # ------------------------------------------------
            # Rank sequence must be contiguous.
            # ------------------------------------------------

            ranks = [
                int(x["rank"])
                for x in hypotheses
            ]

            expected_ranks = list(
                range(
                    1,
                    max_rank + 1,
                )
            )

            if ranks != expected_ranks:

                raise RuntimeError(
                    f"{sid}: rank sequence mismatch\n"
                    f"actual={ranks}\n"
                    f"expected={expected_ranks}"
                )

            out = {
                "sample_id":
                    sid,

                "precursor_mz":
                    precursor_mz,

                "polarity":
                    polarity,

                "peaks":
                    peaks,

                "formula_hypotheses":
                    hypotheses,
            }

            fout.write(
                json.dumps(
                    out,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )

    # ========================================================
    # Final checks
    # ========================================================

    if rank15_formula_check_fail != 0:
        raise RuntimeError(
            "Existing Rank1-5 differs from "
            "Exp1 after normalization: "
            f"{rank15_formula_check_fail}"
        )

    if total_hypotheses != 175529:
        raise RuntimeError(
            f"Top-10 hypothesis count="
            f"{total_hypotheses}, "
            f"expected 175529"
        )

    expected_rank_counts = {
        1: 17556,
        2: 17556,
        3: 17556,
        4: 17556,
        5: 17555,
        6: 17555,
        7: 17555,
        8: 17548,
        9: 17546,
        10: 17546,
    }

    actual_rank_counts = {
        int(k): int(v)
        for k, v in sorted(
            rank_counts.items()
        )
    }

    if actual_rank_counts != expected_rank_counts:

        raise RuntimeError(
            "Rank count mismatch:\n"
            f"actual={actual_rank_counts}\n"
            f"expected={expected_rank_counts}"
        )

    stats = {
        "samples":
            len(sample_order),

        "total_formula_hypotheses":
            total_hypotheses,

        "samples_with_less_than_10":
            samples_with_lt10,

        "rank_counts":
            actual_rank_counts,

        "rank1_5_formula_check_fail":
            rank15_formula_check_fail,

        "formula_confidence_policy":
            "None, identical to existing Top-5",

        "rank1_5_source":
            str(
                top5_path.resolve()
            ),

        "rank6_10_source":
            str(
                exp1_path.resolve()
            ),

        "observable_input_source":
            str(
                top5_path.resolve()
            ),
    }

    stats_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    stats_path.write_text(
        json.dumps(
            stats,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "=" * 80
    )

    print(
        "PREDICTED FORMULA TOP-10 INPUT"
    )

    print(
        "=" * 80
    )

    print(
        "Samples                :",
        len(sample_order),
    )

    print(
        "Formula hypotheses     :",
        total_hypotheses,
    )

    print(
        "Samples with <10       :",
        samples_with_lt10,
    )

    print(
        "Rank1-5 check failures :",
        rank15_formula_check_fail,
    )

    print(
        "Confidence policy      : None"
    )

    print()

    print(
        "Rank counts:"
    )

    for rank in range(1, 11):

        print(
            f"  Rank {rank:2d}: "
            f"{rank_counts[rank]}"
        )

    print()

    print(
        "Output:",
        output_path.resolve(),
    )

    print(
        "Stats :",
        stats_path.resolve(),
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()
