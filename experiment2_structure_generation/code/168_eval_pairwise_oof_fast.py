#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


def read_records(path):
    text = Path(path).read_text(encoding="utf-8").strip()

    if not text:
        return []

    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
    except json.JSONDecodeError:
        pass

    return [
        json.loads(line)
        for line in text.splitlines()
        if line.strip()
    ]


def true_smiles(row):
    mol = row.get("molecule") or {}

    for x in (
        mol.get("smiles"),
        row.get("true_smiles"),
        row.get("smiles"),
    ):
        if isinstance(x, str) and x:
            return x

    return None


def ik2d(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            return None

        return Chem.MolToInchiKey(
            mol
        ).split("-")[0]

    except Exception:
        return None


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
        "--with-truth",
        required=True
    )

    args = ap.parse_args()

    preds = read_records(
        args.predictions
    )

    truths = read_records(
        args.massspecgym
    )

    truth_by_id = {
        str(x["identifier"]):
            true_smiles(x)
        for x in truths
    }

    fpgen = (
        rdFingerprintGenerator
        .GetMorganGenerator(
            radius=2,
            fpSize=2048
        )
    )

    n = 0

    hit1 = 0
    hit10 = 0

    sim1_sum = 0.0
    sim10_sum = 0.0

    attached = []

    for row in preds:

        sid = str(
            row["identifier"]
        )

        truth = truth_by_id.get(
            sid
        )

        if not truth:
            raise RuntimeError(
                f"Missing truth: {sid}"
            )

        truth_mol = Chem.MolFromSmiles(
            truth
        )

        truth_key = ik2d(
            truth
        )

        truth_fp = (
            fpgen.GetFingerprint(
                truth_mol
            )
            if truth_mol is not None
            else None
        )

        predictions = (
            row.get("predictions")
            or []
        )[:10]

        keys = []
        sims = []

        for smiles in predictions:

            mol = Chem.MolFromSmiles(
                smiles
            )

            keys.append(
                ik2d(smiles)
            )

            if (
                mol is None
                or truth_fp is None
            ):
                sims.append(0.0)

            else:
                fp = fpgen.GetFingerprint(
                    mol
                )

                sims.append(
                    DataStructs.TanimotoSimilarity(
                        truth_fp,
                        fp
                    )
                )

        this_hit1 = bool(
            keys
            and keys[0] is not None
            and keys[0] == truth_key
        )

        this_hit10 = any(
            k is not None
            and k == truth_key
            for k in keys
        )

        hit1 += int(this_hit1)
        hit10 += int(this_hit10)

        # Missing prediction follows the previous protocol:
        # similarity = 0.
        sim1_sum += (
            sims[0]
            if sims
            else 0.0
        )

        sim10_sum += (
            max(sims)
            if sims
            else 0.0
        )

        out = dict(row)

        out["true_smiles"] = truth

        attached.append(out)

        n += 1

    metrics = {
        "num_samples":
            n,

        "top1_hits":
            hit1,

        "top1_accuracy":
            hit1 / n,

        "top10_hits":
            hit10,

        "top10_accuracy":
            hit10 / n,

        "ecfp4_tanimoto_at1":
            sim1_sum / n,

        "ecfp4_tanimoto_at10":
            sim10_sum / n,
    }

    Path(args.output).write_text(
        json.dumps(
            metrics,
            indent=2
        ),
        encoding="utf-8"
    )

    with Path(
        args.with_truth
    ).open(
        "w",
        encoding="utf-8"
    ) as f:

        for row in attached:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False
                )
                + "\n"
            )

    print()
    print("=" * 72)
    print("PAIRWISE OOF FAST METRICS")
    print("=" * 72)

    print(
        f"Top1   : {hit1}/{n} "
        f"= {hit1/n:.2%}"
    )

    print(
        f"Top10  : {hit10}/{n} "
        f"= {hit10/n:.2%}"
    )

    print(
        f"ECFP4@1 : "
        f"{sim1_sum/n:.4f}"
    )

    print(
        f"ECFP4@10: "
        f"{sim10_sum/n:.4f}"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()
