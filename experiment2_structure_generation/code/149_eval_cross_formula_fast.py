#!/usr/bin/env python3

import argparse
import json

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem


def mol(s):
    if not s:
        return None

    try:
        return Chem.MolFromSmiles(s)
    except Exception:
        return None


def key2d(m):
    if m is None:
        return None

    try:
        return (
            Chem.MolToInchiKey(m)
            .split("-")[0]
        )
    except Exception:
        return None


def fp(m):
    if m is None:
        return None

    return (
        AllChem
        .GetMorganFingerprintAsBitVect(
            m,
            radius=2,
            nBits=2048,
        )
    )


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--input",
        required=True
    )

    args = ap.parse_args()

    n = 0
    top1 = 0
    top10 = 0

    tan1_sum = 0.0
    tan10_sum = 0.0

    with open(
        args.input,
        encoding="utf-8"
    ) as f:

        for line in f:

            if not line.strip():
                continue

            row = json.loads(line)

            truth = mol(
                row["true_smiles"]
            )

            tkey = key2d(truth)
            tfp = fp(truth)

            preds = (
                row.get("predictions")
                or []
            )[:10]

            infos = []

            for smi in preds:

                pm = mol(smi)

                pkey = key2d(pm)
                pfp = fp(pm)

                exact = (
                    tkey is not None
                    and pkey == tkey
                )

                tan = (
                    DataStructs
                    .TanimotoSimilarity(
                        tfp,
                        pfp
                    )
                    if (
                        tfp is not None
                        and pfp is not None
                    )
                    else 0.0
                )

                infos.append(
                    (
                        exact,
                        tan
                    )
                )

            n += 1

            top1 += int(
                bool(infos)
                and infos[0][0]
            )

            top10 += int(
                any(
                    x[0]
                    for x in infos
                )
            )

            tan1_sum += (
                infos[0][1]
                if infos
                else 0.0
            )

            tan10_sum += (
                max(
                    (
                        x[1]
                        for x in infos
                    ),
                    default=0.0
                )
            )

    print(
        "N:",
        n
    )

    print(
        "Top-1 Accuracy:",
        top1 / n
    )

    print(
        "Top-10 Accuracy:",
        top10 / n
    )

    print(
        "ECFP4 Tanimoto@1:",
        tan1_sum / n
    )

    print(
        "ECFP4 Tanimoto@10:",
        tan10_sum / n
    )


if __name__ == "__main__":
    main()
