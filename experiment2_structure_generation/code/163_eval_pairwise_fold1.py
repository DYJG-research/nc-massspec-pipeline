#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


THRESHOLDS = [
    0.25,
    0.50,
    0.75,
    1.00,
    1.25,
    1.50,
    1.75,
    2.00,
    2.25,
    2.50,
    2.75,
    3.00,
    3.50,
    4.00,
    4.50,
]


def read_jsonl(path):
    rows = []

    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                rows.append(json.loads(line))
            except Exception as exc:
                raise RuntimeError(
                    f"{path}:{line_no}: invalid JSON: {exc}"
                ) from exc

    return rows


def rank_candidates(rows):
    """
    Reproduce the frozen XGB OOF ranking.
    """

    ordered = sorted(
        rows,
        key=lambda r: (
            -float(r["oof_score"]),
            int(r.get("formula_rank", 999)),
            int(r.get("branch_route_rank", 999)),
            str(r.get("candidate_id", "")),
        ),
    )

    output = []
    seen_smiles = set()

    for row in ordered:
        smiles = str(row.get("smiles") or "")

        if not smiles:
            continue

        if smiles in seen_smiles:
            continue

        seen_smiles.add(smiles)
        output.append(row)

    return output


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--predictions",
        required=True,
        help="Pairwise verifier prediction JSONL.",
    )

    ap.add_argument(
        "--scores",
        required=True,
        help="Frozen XGB OOF candidate scores.",
    )

    ap.add_argument(
        "--fold",
        type=int,
        default=1,
    )

    ap.add_argument(
        "--output",
        required=True,
    )

    args = ap.parse_args()

    # ============================================================
    # Pairwise LLM outputs
    # ============================================================

    prediction_rows = read_jsonl(
        args.predictions
    )

    predictions = {
        str(row["sample_id"]): row
        for row in prediction_rows
    }

    if len(predictions) != len(prediction_rows):
        raise RuntimeError(
            "Duplicate sample_id found in pairwise predictions."
        )

    # ============================================================
    # Frozen XGB scores
    # ============================================================

    by_sample = defaultdict(list)

    for row in read_jsonl(args.scores):

        if int(row["fold"]) != args.fold:
            continue

        by_sample[
            str(row["sample_id"])
        ].append(row)

    # ============================================================
    # Reconstruct Fold-X XGB ranking + truth
    # ============================================================

    samples = []

    baseline_hits = 0
    nonempty = 0

    truth_in_top5 = 0
    truth_in_top10 = 0

    for sid, candidate_rows in by_sample.items():

        ranked = rank_candidates(
            candidate_rows
        )

        if not ranked:
            continue

        nonempty += 1

        top10 = ranked[:10]

        truth_id = None
        truth_rank = None

        for rank, row in enumerate(
            top10,
            1,
        ):
            if int(row.get("label", 0)) == 1:
                truth_id = str(
                    row["candidate_id"]
                )
                truth_rank = rank
                break

        if (
            truth_rank is not None
            and truth_rank <= 5
        ):
            truth_in_top5 += 1

        if truth_rank is not None:
            truth_in_top10 += 1

        baseline_id = str(
            top10[0]["candidate_id"]
        )

        baseline_correct = (
            truth_id is not None
            and baseline_id == truth_id
        )

        baseline_hits += int(
            baseline_correct
        )

        if len(top10) >= 2:
            margin = (
                float(top10[0]["oof_score"])
                -
                float(top10[1]["oof_score"])
            )
        else:
            margin = float("inf")

        pred = predictions.get(sid)

        samples.append(
            {
                "sample_id":
                    sid,

                "truth_id":
                    truth_id,

                "truth_rank":
                    truth_rank,

                "baseline_id":
                    baseline_id,

                "baseline_correct":
                    baseline_correct,

                "margin":
                    margin,

                "prediction":
                    pred,
            }
        )

    n = len(samples)

    # ============================================================
    # Basic prediction diagnostics
    # ============================================================

    pred_triggered = len(predictions)

    confirmed_count_distribution = defaultdict(int)

    for row in prediction_rows:
        confirmed_count_distribution[
            int(
                row.get(
                    "num_confirmed_challengers",
                    0
                )
            )
        ] += 1

    # ============================================================
    # Evaluate two conservative replacement policies
    # ============================================================

    policies = [
        "final_unique_confirmed",
        "final_lowest_rank_confirmed",
    ]

    all_results = {}

    for policy in policies:

        gate_rows = []

        for threshold in THRESHOLDS:

            hits = 0

            triggered = 0
            changed = 0

            fixed = 0
            broken = 0

            no_prediction_triggered = 0

            for row in samples:

                final_id = row[
                    "baseline_id"
                ]

                # --------------------------------------------
                # Gate
                # --------------------------------------------

                if row["margin"] <= threshold:

                    triggered += 1

                    pred = row[
                        "prediction"
                    ]

                    if pred is None:
                        no_prediction_triggered += 1

                    else:
                        proposed_id = pred.get(
                            policy
                        )

                        if proposed_id:
                            final_id = str(
                                proposed_id
                            )

                if (
                    final_id
                    != row["baseline_id"]
                ):
                    changed += 1

                final_correct = (
                    row["truth_id"] is not None
                    and final_id
                    == row["truth_id"]
                )

                hits += int(
                    final_correct
                )

                # --------------------------------------------
                # Fixed / broken
                # --------------------------------------------

                if (
                    not row["baseline_correct"]
                    and final_correct
                ):
                    fixed += 1

                if (
                    row["baseline_correct"]
                    and not final_correct
                ):
                    broken += 1

            delta = (
                hits
                - baseline_hits
            )

            assert delta == (
                fixed - broken
            ), (
                f"Internal inconsistency at threshold "
                f"{threshold}: "
                f"delta={delta}, "
                f"fix-break={fixed-broken}"
            )

            gate_rows.append(
                {
                    "threshold":
                        threshold,

                    "triggered":
                        triggered,

                    "trigger_rate":
                        (
                            triggered / n
                            if n else 0.0
                        ),

                    "changed":
                        changed,

                    "change_rate":
                        (
                            changed / n
                            if n else 0.0
                        ),

                    "fixed_errors":
                        fixed,

                    "broken_correct":
                        broken,

                    "net_gain":
                        fixed - broken,

                    "top1_hits":
                        hits,

                    "top1_accuracy":
                        (
                            hits / n
                            if n else 0.0
                        ),

                    "delta_hits":
                        delta,

                    "delta_pp":
                        (
                            100.0
                            * delta
                            / n
                            if n else 0.0
                        ),

                    "no_prediction_triggered":
                        no_prediction_triggered,
                }
            )

        # Best = max accuracy, then fewer modifications,
        # then smaller threshold.
        best = max(
            gate_rows,
            key=lambda r: (
                r["top1_hits"],
                -r["changed"],
                -r["threshold"],
            ),
        )

        all_results[
            policy
        ] = {
            "best":
                best,

            "gate_sweep":
                gate_rows,
        }

    # ============================================================
    # Summary
    # ============================================================

    summary = {
        "fold":
            args.fold,

        "num_nonempty_samples":
            n,

        "pairwise_prediction_samples":
            pred_triggered,

        "baseline_xgb_top1_hits":
            baseline_hits,

        "baseline_xgb_top1_accuracy":
            (
                baseline_hits / n
                if n else 0.0
            ),

        "truth_in_top5":
            truth_in_top5,

        "truth_in_top10":
            truth_in_top10,

        "confirmed_challenger_count_distribution":
            dict(
                sorted(
                    confirmed_count_distribution.items()
                )
            ),

        "policies":
            all_results,
    }

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ============================================================
    # Human-readable output
    # ============================================================

    print()
    print("=" * 112)
    print(
        f"PAIRWISE VERIFIER V2, FOLD-{args.fold}"
    )
    print("=" * 112)

    print(
        "Samples                    :",
        n
    )

    print(
        "Pairwise-predicted samples :",
        pred_triggered
    )

    print(
        "XGB baseline               :",
        f"{baseline_hits}/{n}",
        f"= {baseline_hits/n:.2%}"
    )

    print(
        "Truth in XGB Top-5         :",
        truth_in_top5
    )

    print(
        "Truth in XGB Top-10        :",
        truth_in_top10
    )

    print(
        "Confirmed challenger counts:",
        dict(
            sorted(
                confirmed_count_distribution.items()
            )
        )
    )

    for policy in policies:

        print()
        print("=" * 112)
        print(
            "POLICY:",
            policy
        )
        print("=" * 112)

        print(
            f"{'Thr':>6}"
            f"{'Trig':>8}"
            f"{'Change':>9}"
            f"{'Fix':>7}"
            f"{'Break':>8}"
            f"{'Net':>7}"
            f"{'Hits':>8}"
            f"{'Top1':>10}"
            f"{'Delta':>8}"
            f"{'Δpp':>9}"
        )

        print("-" * 112)

        for r in all_results[
            policy
        ]["gate_sweep"]:

            print(
                f"{r['threshold']:>6.2f}"
                f"{r['triggered']:>8}"
                f"{r['changed']:>9}"
                f"{r['fixed_errors']:>7}"
                f"{r['broken_correct']:>8}"
                f"{r['net_gain']:>7}"
                f"{r['top1_hits']:>8}"
                f"{r['top1_accuracy']*100:>9.2f}%"
                f"{r['delta_hits']:>8}"
                f"{r['delta_pp']:>8.2f}"
            )

        best = all_results[
            policy
        ]["best"]

        print("-" * 112)

        print(
            "BEST:",
            f"threshold={best['threshold']}",
            f"triggered={best['triggered']}",
            f"changed={best['changed']}",
            f"fix={best['fixed_errors']}",
            f"break={best['broken_correct']}",
            f"net={best['net_gain']}",
            f"hits={best['top1_hits']}/{n}",
            f"Top1={best['top1_accuracy']:.2%}",
            f"delta={best['delta_hits']}",
            f"delta_pp={best['delta_pp']:.2f}",
        )

    print()
    print("=" * 112)

    # Predeclared pilot decision.
    best_overall = max(
        (
            all_results[p][
                "best"
            ]
            for p in policies
        ),
        key=lambda r:
            r["top1_hits"],
    )

    if best_overall[
        "delta_hits"
    ] >= 5:

        print(
            "PILOT DECISION: GO"
        )

        print(
            "Pairwise verifier reached the "
            "predeclared Fold-1 threshold "
            "of Delta >= +5 hits."
        )

    else:

        print(
            "PILOT DECISION: NO-GO"
        )

        print(
            "Pairwise verifier did not reach "
            "the predeclared Fold-1 threshold "
            "of Delta >= +5 hits."
        )

    print("=" * 112)

    print(
        "[OK] Metrics:",
        output_path
    )


if __name__ == "__main__":
    main()
