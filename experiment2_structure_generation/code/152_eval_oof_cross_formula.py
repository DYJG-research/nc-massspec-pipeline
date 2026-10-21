#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")


def load_rows(path):
    text = Path(path).read_text(
        encoding="utf-8"
    ).strip()

    if not text:
        return []

    if text.startswith("["):
        return json.loads(text)

    return [
        json.loads(line)
        for line in text.splitlines()
        if line.strip()
    ]


def mol(smiles):
    if not smiles:
        return None

    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def key2d(m):
    if m is None:
        return None

    try:
        return Chem.MolToInchiKey(m).split("-")[0]
    except Exception:
        return None


def fingerprint(m):
    if m is None:
        return None

    return AllChem.GetMorganFingerprintAsBitVect(
        m,
        radius=2,
        nBits=2048,
    )


def evaluate(prediction_path, truth):
    rows = load_rows(prediction_path)

    n = 0
    nonempty = 0

    top1_hits = 0
    top10_hits = 0

    ecfp1_sum = 0.0
    ecfp10_sum = 0.0

    per_sample = []

    for row in rows:
        sid = str(row["identifier"])

        if sid not in truth:
            raise RuntimeError(
                f"Missing truth: {sid}"
            )

        truth_mol = mol(
            truth[sid]
        )

        truth_key = key2d(
            truth_mol
        )

        truth_fp = fingerprint(
            truth_mol
        )

        predictions = (
            row.get("predictions") or []
        )[:10]

        n += 1

        if predictions:
            nonempty += 1

        exact = []
        similarities = []

        for smi in predictions:
            pmol = mol(smi)

            pkey = key2d(
                pmol
            )

            pfp = fingerprint(
                pmol
            )

            exact_hit = bool(
                truth_key is not None
                and pkey is not None
                and truth_key == pkey
            )

            if (
                truth_fp is not None
                and pfp is not None
            ):
                similarity = (
                    DataStructs
                    .TanimotoSimilarity(
                        truth_fp,
                        pfp,
                    )
                )
            else:
                similarity = 0.0

            exact.append(
                exact_hit
            )

            similarities.append(
                similarity
            )

        top1 = bool(
            exact
            and exact[0]
        )

        top10 = any(
            exact
        )

        tan1 = (
            similarities[0]
            if similarities
            else 0.0
        )

        tan10 = max(
            similarities,
            default=0.0,
        )

        top1_hits += int(
            top1
        )

        top10_hits += int(
            top10
        )

        ecfp1_sum += tan1
        ecfp10_sum += tan10

        per_sample.append(
            {
                "identifier":
                    sid,

                "top1_exact":
                    top1,

                "top10_exact":
                    top10,

                "ecfp4_at1":
                    tan1,

                "ecfp4_at10":
                    tan10,

                "num_predictions":
                    len(
                        predictions
                    ),
            }
        )

    return {
        "num_samples":
            n,

        "coverage":
            (
                nonempty / n
                if n
                else 0.0
            ),

        "top1_hits":
            top1_hits,

        "top10_hits":
            top10_hits,

        "top1_accuracy":
            (
                top1_hits / n
                if n
                else 0.0
            ),

        "top10_accuracy":
            (
                top10_hits / n
                if n
                else 0.0
            ),

        "ecfp4_tanimoto_at1":
            (
                ecfp1_sum / n
                if n
                else 0.0
            ),

        "ecfp4_tanimoto_at10":
            (
                ecfp10_sum / n
                if n
                else 0.0
            ),

        "per_sample":
            per_sample,
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--predictions",
        required=True
    )

    ap.add_argument(
        "--massspecgym",
        required=True
    )

    ap.add_argument(
        "--output",
        required=True
    )

    ap.add_argument(
        "--baseline",
        default=""
    )

    args = ap.parse_args()

    mass = load_rows(
        args.massspecgym
    )

    truth = {
        str(x["identifier"]):
            str(
                x["molecule"]["smiles"]
            )
        for x in mass
    }

    result = evaluate(
        args.predictions,
        truth,
    )

    output = {
        "oof": {
            k: v
            for k, v in result.items()
            if k != "per_sample"
        },

        "per_sample":
            result["per_sample"],
    }

    baseline_result = None
    deltas = None

    if args.baseline:
        baseline_result = evaluate(
            args.baseline,
            truth,
        )

        deltas = {
            "top1_accuracy":
                (
                    result["top1_accuracy"]
                    - baseline_result[
                        "top1_accuracy"
                    ]
                ),

            "top10_accuracy":
                (
                    result["top10_accuracy"]
                    - baseline_result[
                        "top10_accuracy"
                    ]
                ),

            "ecfp4_tanimoto_at1":
                (
                    result[
                        "ecfp4_tanimoto_at1"
                    ]
                    - baseline_result[
                        "ecfp4_tanimoto_at1"
                    ]
                ),

            "ecfp4_tanimoto_at10":
                (
                    result[
                        "ecfp4_tanimoto_at10"
                    ]
                    - baseline_result[
                        "ecfp4_tanimoto_at10"
                    ]
                ),
        }

        output["baseline"] = {
            k: v
            for k, v
            in baseline_result.items()
            if k != "per_sample"
        }

        output[
            "delta_vs_baseline"
        ] = deltas

    Path(
        args.output
    ).write_text(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8"
    )

    print()
    print("=" * 78)

    print(
        "XGB OOF CROSS-FORMULA RANKING"
    )

    print("=" * 78)

    print(
        f"N                  : "
        f"{result['num_samples']}"
    )

    print(
        f"Coverage           : "
        f"{result['coverage'] * 100:.2f}%"
    )

    print(
        f"Top-1 Accuracy     : "
        f"{result['top1_accuracy'] * 100:.2f}% "
        f"({result['top1_hits']}/"
        f"{result['num_samples']})"
    )

    print(
        f"Top-10 Accuracy    : "
        f"{result['top10_accuracy'] * 100:.2f}% "
        f"({result['top10_hits']}/"
        f"{result['num_samples']})"
    )

    print(
        f"ECFP4 Tanimoto@1   : "
        f"{result['ecfp4_tanimoto_at1']:.4f}"
    )

    print(
        f"ECFP4 Tanimoto@10  : "
        f"{result['ecfp4_tanimoto_at10']:.4f}"
    )

    if deltas is not None:
        print()

        print(
            "Compared with Formula + Branch fusion:"
        )

        print(
            f"Delta Top-1        : "
            f"{deltas['top1_accuracy'] * 100:+.2f} pp"
        )

        print(
            f"Delta Top-10       : "
            f"{deltas['top10_accuracy'] * 100:+.2f} pp"
        )

        print(
            f"Delta ECFP4@1      : "
            f"{deltas['ecfp4_tanimoto_at1']:+.4f}"
        )

        print(
            f"Delta ECFP4@10     : "
            f"{deltas['ecfp4_tanimoto_at10']:+.4f}"
        )

    print("=" * 78)


if __name__ == "__main__":
    main()
