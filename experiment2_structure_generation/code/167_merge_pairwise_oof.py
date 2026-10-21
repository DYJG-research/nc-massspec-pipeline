#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def rank_candidates(rows):
    ordered = sorted(
        rows,
        key=lambda r: (
            -float(r["oof_score"]),
            int(r.get("formula_rank", 999)),
            int(r.get("branch_route_rank", 999)),
            str(r.get("candidate_id", "")),
        ),
    )

    out = []
    seen = set()

    for row in ordered:
        smi = str(row.get("smiles") or "")

        if not smi or smi in seen:
            continue

        seen.add(smi)
        out.append(row)

    return out


def load_pairwise(v2_dir):
    paths = {
        1: v2_dir / "fold1_pilot/fold1_pairwise_predictions.jsonl",
        2: v2_dir / "fold2_oof/fold2_pairwise_predictions.jsonl",
        3: v2_dir / "fold3_oof/fold3_pairwise_predictions.jsonl",
        4: v2_dir / "fold4_oof/fold4_pairwise_predictions.jsonl",
        5: v2_dir / "fold5_oof/fold5_pairwise_predictions.jsonl",
    }

    result = {}

    for fold, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)

        for row in read_jsonl(path):
            sid = str(row["sample_id"])

            if sid in result:
                raise RuntimeError(
                    f"Duplicate pairwise prediction: {sid}"
                )

            result[sid] = row

    return result


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--scores", required=True)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--v2-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--summary", required=True)

    ap.add_argument(
        "--threshold",
        type=float,
        default=0.50,
    )

    ap.add_argument(
        "--policy",
        default="final_lowest_rank_confirmed",
    )

    args = ap.parse_args()

    scores = read_jsonl(args.scores)
    samples = read_jsonl(args.samples)

    pairwise = load_pairwise(
        Path(args.v2_dir)
    )

    by_sample = defaultdict(list)

    for row in scores:
        by_sample[
            str(row["sample_id"])
        ].append(row)

    sample_ids = [
        str(row["sample_id"])
        for row in samples
    ]

    n_total = len(sample_ids)

    baseline_hits = 0
    final_hits = 0

    baseline_top10_hits = 0
    final_top10_hits = 0

    triggered = 0
    changed = 0
    fixed = 0
    broken = 0
    empty = 0

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as fout:

        for sid in sample_ids:

            ranked = rank_candidates(
                by_sample.get(sid, [])
            )

            # Preserve the 37 empty samples in the
            # formal 2643-sample denominator.
            if not ranked:
                empty += 1

                fout.write(
                    json.dumps(
                        {
                            "identifier": sid,
                            "sample_id": sid,
                            "predictions": [],
                            "candidate_ids": [],
                            "pairwise_applied": False,
                            "empty_prediction": True,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                continue

            top10 = ranked[:10]

            truth_ids = {
                str(row["candidate_id"])
                for row in top10
                if int(row.get("label", 0)) == 1
            }

            baseline_id = str(
                top10[0]["candidate_id"]
            )

            baseline_correct = (
                baseline_id in truth_ids
            )

            baseline_hits += int(
                baseline_correct
            )

            baseline_top10_hits += int(
                bool(truth_ids)
            )

            if len(top10) >= 2:
                margin = (
                    float(top10[0]["oof_score"])
                    -
                    float(top10[1]["oof_score"])
                )
            else:
                margin = float("inf")

            final_id = baseline_id
            applied = False

            if margin <= args.threshold:
                triggered += 1

                pred = pairwise.get(sid)

                if pred is not None:
                    proposed = pred.get(
                        args.policy
                    )

                    valid_ids = {
                        str(x["candidate_id"])
                        for x in top10
                    }

                    if (
                        proposed is not None
                        and str(proposed) in valid_ids
                    ):
                        final_id = str(proposed)
                        applied = True

            if final_id != baseline_id:
                changed += 1

            final_correct = (
                final_id in truth_ids
            )

            final_hits += int(
                final_correct
            )

            if (
                not baseline_correct
                and final_correct
            ):
                fixed += 1

            if (
                baseline_correct
                and not final_correct
            ):
                broken += 1

            # Same Top-10 set, only move selected candidate to rank 1.
            final_rows = list(top10)

            if final_id != baseline_id:
                selected = next(
                    x
                    for x in final_rows
                    if str(x["candidate_id"]) == final_id
                )

                final_rows = (
                    [selected]
                    +
                    [
                        x
                        for x in final_rows
                        if str(x["candidate_id"]) != final_id
                    ]
                )

            final_top10_hits += int(
                bool(truth_ids)
            )

            fout.write(
                json.dumps(
                    {
                        "identifier": sid,
                        "sample_id": sid,
                        "fold": int(top10[0]["fold"]),
                        "xgb_margin_top1_top2": margin,
                        "pairwise_gate_threshold": args.threshold,
                        "pairwise_policy": args.policy,
                        "pairwise_applied": applied,
                        "empty_prediction": False,
                        "predictions": [
                            x["smiles"]
                            for x in final_rows
                        ],
                        "candidate_ids": [
                            x["candidate_id"]
                            for x in final_rows
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    summary = {
        "num_samples": n_total,
        "num_empty_predictions": empty,

        "threshold": args.threshold,
        "policy": args.policy,

        "baseline_top1_hits": baseline_hits,
        "baseline_top1_accuracy":
            baseline_hits / n_total,

        "final_top1_hits": final_hits,
        "final_top1_accuracy":
            final_hits / n_total,

        "delta_hits":
            final_hits - baseline_hits,

        "delta_pp":
            100.0
            * (final_hits - baseline_hits)
            / n_total,

        "fixed_errors": fixed,
        "broken_correct": broken,

        "triggered": triggered,
        "changed": changed,

        "baseline_top10_hits":
            baseline_top10_hits,

        "final_top10_hits":
            final_top10_hits,

        "final_top10_accuracy":
            final_top10_hits / n_total,
    }

    Path(args.summary).write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("FINAL 5-FOLD OOF PAIRWISE MERGE")
    print("=" * 80)

    print("Total samples :", n_total)
    print("Empty         :", empty)

    print(
        "XGB Top1      :",
        f"{baseline_hits}/{n_total}",
        f"= {baseline_hits/n_total:.2%}"
    )

    print(
        "XGB + LLM     :",
        f"{final_hits}/{n_total}",
        f"= {final_hits/n_total:.2%}"
    )

    print("Fix           :", fixed)
    print("Break         :", broken)
    print("Delta hits    :", final_hits - baseline_hits)
    print("Delta pp      :", f"{summary['delta_pp']:.2f}")

    print(
        "Top10         :",
        f"{final_top10_hits}/{n_total}",
        f"= {final_top10_hits/n_total:.2%}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()
