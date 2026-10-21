#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import urllib.request
from pathlib import Path


SEL_RE = re.compile(
    r"<selection>\s*([AB])\s*</selection>",
    re.I,
)


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def make_user(spectrum, block_a, block_b):
    return (
        f"{spectrum}\n\n"
        "Candidate A\n"
        "-----------\n"
        f"{block_a}\n\n"
        "Candidate B\n"
        "-----------\n"
        f"{block_b}\n\n"
        "Which candidate is more consistent with the observed "
        "MS/MS evidence?\n"
        "Output only <selection>A</selection> or "
        "<selection>B</selection>."
    )


def api_call(
    base_url,
    model,
    system,
    user,
    timeout,
):
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": system,
            },
            {
                "role": "user",
                "content": user,
            },
        ],
        "temperature": 0,
        "max_tokens": 32,
        "stream": False,
    }

    req = urllib.request.Request(
        base_url.rstrip("/")
        + "/chat/completions",
        data=json.dumps(
            payload,
            ensure_ascii=False,
        ).encode("utf-8"),
        headers={
            "Content-Type": "application/json"
        },
        method="POST",
    )

    with urllib.request.urlopen(
        req,
        timeout=timeout,
    ) as response:
        obj = json.loads(
            response.read().decode("utf-8")
        )

    text = str(
        obj["choices"][0]["message"]["content"]
        or ""
    ).strip()

    match = SEL_RE.search(text)

    label = (
        match.group(1).upper()
        if match
        else None
    )

    return label, text


def first_orientation(
    sample_id,
    challenger_id,
):
    text = (
        f"{sample_id}::{challenger_id}"
    )

    digest = hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()

    if int(digest[:8], 16) % 2 == 0:
        return "baseline_A"

    return "baseline_B"


def compare_once(
    base_url,
    model,
    system,
    spectrum,
    baseline,
    challenger,
    orientation,
    timeout,
):
    if orientation == "baseline_A":
        block_a = baseline["visible_block"]
        block_b = challenger["visible_block"]

        label_map = {
            "A": baseline["candidate_id"],
            "B": challenger["candidate_id"],
        }

    else:
        block_a = challenger["visible_block"]
        block_b = baseline["visible_block"]

        label_map = {
            "A": challenger["candidate_id"],
            "B": baseline["candidate_id"],
        }

    user = make_user(
        spectrum,
        block_a,
        block_b,
    )

    label, raw = api_call(
        base_url,
        model,
        system,
        user,
        timeout,
    )

    candidate_id = (
        label_map.get(label)
        if label
        else None
    )

    return {
        "orientation": orientation,
        "selection_label": label,
        "selected_candidate_id": candidate_id,
        "parse_ok": candidate_id is not None,
        "raw_output": raw,
    }


def process_sample(
    row,
    base_url,
    model,
    challenger_top_k,
    timeout,
):
    candidates = row["candidates"]

    baseline = candidates[0]

    # challenger_top_k=5 means:
    # XGB Rank1 vs Rank2/3/4/5
    challengers = candidates[
        1:challenger_top_k
    ]

    system = row["system_prompt"]
    spectrum = row["spectrum_block"]

    comparisons = []
    confirmed = []

    for challenger in challengers:

        orientation = first_orientation(
            row["sample_id"],
            challenger["candidate_id"],
        )

        first = compare_once(
            base_url,
            model,
            system,
            spectrum,
            baseline,
            challenger,
            orientation,
            timeout,
        )

        result = {
            "challenger_candidate_id":
                challenger["candidate_id"],

            "challenger_xgb_rank":
                challenger["xgb_rank"],

            "first":
                first,

            "reverse":
                None,

            "confirmed_challenger":
                False,
        }

        # Only if challenger wins the first comparison
        # do we reverse A/B and verify again.
        if (
            first["selected_candidate_id"]
            == challenger["candidate_id"]
        ):

            reverse_orientation = (
                "baseline_B"
                if orientation == "baseline_A"
                else "baseline_A"
            )

            second = compare_once(
                base_url,
                model,
                system,
                spectrum,
                baseline,
                challenger,
                reverse_orientation,
                timeout,
            )

            result["reverse"] = second

            # Challenger must win BOTH orientations.
            if (
                second["selected_candidate_id"]
                == challenger["candidate_id"]
            ):
                result[
                    "confirmed_challenger"
                ] = True

                confirmed.append(
                    {
                        "candidate_id":
                            challenger[
                                "candidate_id"
                            ],

                        "xgb_rank":
                            challenger[
                                "xgb_rank"
                            ],
                    }
                )

        comparisons.append(result)

    # Policy 1:
    # only replace baseline if exactly one
    # challenger is confirmed.
    if len(confirmed) == 1:
        final_unique = (
            confirmed[0]["candidate_id"]
        )
    else:
        final_unique = (
            baseline["candidate_id"]
        )

    # Policy 2:
    # if multiple challengers are confirmed,
    # use the one that originally ranked highest.
    if confirmed:
        best = min(
            confirmed,
            key=lambda x: x["xgb_rank"],
        )

        final_lowest_rank = (
            best["candidate_id"]
        )
    else:
        final_lowest_rank = (
            baseline["candidate_id"]
        )

    return {
        "sample_id":
            row["sample_id"],

        "fold":
            row["fold"],

        "xgb_margin_top1_top2":
            row["xgb_margin_top1_top2"],

        "baseline_candidate_id":
            baseline["candidate_id"],

        "num_confirmed_challengers":
            len(confirmed),

        "confirmed_challengers":
            confirmed,

        "final_unique_confirmed":
            final_unique,

        "final_lowest_rank_confirmed":
            final_lowest_rank,

        "comparisons":
            comparisons,
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--context",
        required=True,
    )

    ap.add_argument(
        "--output",
        required=True,
    )

    ap.add_argument(
        "--fold",
        type=int,
        default=1,
    )

    ap.add_argument(
        "--max-margin",
        type=float,
        default=4.5,
    )

    ap.add_argument(
        "--challenger-top-k",
        type=int,
        default=5,
    )

    ap.add_argument(
        "--base-url",
        default="http://127.0.0.1:19999/v1",
    )

    ap.add_argument(
        "--model",
        default="exp3-pairwise-verifier-v2",
    )

    ap.add_argument(
        "--workers",
        type=int,
        default=2,
    )

    ap.add_argument(
        "--timeout",
        type=float,
        default=180.0,
    )

    ap.add_argument(
        "--limit",
        type=int,
        default=0,
    )

    args = ap.parse_args()

    all_rows = read_jsonl(
        args.context
    )

    rows = []

    for row in all_rows:

        if int(row["fold"]) != args.fold:
            continue

        margin = row.get(
            "xgb_margin_top1_top2"
        )

        if margin is None:
            continue

        if float(margin) > args.max_margin:
            continue

        rows.append(row)

    if args.limit > 0:
        rows = rows[:args.limit]

    print("=" * 80)
    print("PAIRWISE VERIFIER API")
    print("=" * 80)

    print(
        "Triggered samples :",
        len(rows),
    )

    print(
        "Fold              :",
        args.fold,
    )

    print(
        "Max margin        :",
        args.max_margin,
    )

    print(
        "Top-K considered  :",
        args.challenger_top_k,
    )

    print(
        "Comparisons/sample:",
        args.challenger_top_k - 1,
    )

    print(
        "Workers           :",
        args.workers,
    )

    print("=" * 80)

    results = []

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.workers
    ) as executor:

        futures = [
            executor.submit(
                process_sample,
                row,
                args.base_url,
                args.model,
                args.challenger_top_k,
                args.timeout,
            )
            for row in rows
        ]

        for idx, future in enumerate(
            concurrent.futures.as_completed(
                futures
            ),
            1,
        ):
            result = future.result()

            results.append(result)

            if (
                idx % 20 == 0
                or idx == len(futures)
            ):
                n_changed = sum(
                    x[
                        "final_unique_confirmed"
                    ]
                    != x[
                        "baseline_candidate_id"
                    ]
                    for x in results
                )

                print(
                    f"[{idx}/{len(futures)}] "
                    f"unique_changed={n_changed}"
                )

    results.sort(
        key=lambda x: x["sample_id"]
    )

    output = Path(args.output)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output.open(
        "w",
        encoding="utf-8",
    ) as f:
        for row in results:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    changed_unique = sum(
        x["final_unique_confirmed"]
        != x["baseline_candidate_id"]
        for x in results
    )

    changed_lowest = sum(
        x["final_lowest_rank_confirmed"]
        != x["baseline_candidate_id"]
        for x in results
    )

    confirmed_counts = {}

    for row in results:
        n = row[
            "num_confirmed_challengers"
        ]

        confirmed_counts[n] = (
            confirmed_counts.get(n, 0)
            + 1
        )

    print()
    print("=" * 80)
    print("FINISHED")
    print("=" * 80)

    print(
        "Samples                    :",
        len(results),
    )

    print(
        "Unique-confirmed changes   :",
        changed_unique,
    )

    print(
        "Lowest-rank policy changes :",
        changed_lowest,
    )

    print(
        "Confirmed-count distribution:",
        confirmed_counts,
    )

    print(
        "Output:",
        output,
    )

    print("=" * 80)


if __name__ == "__main__":
    main()
