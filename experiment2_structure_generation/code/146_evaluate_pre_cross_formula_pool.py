#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.error")
RDLogger.DisableLog("rdApp.warning")


def read_json_or_jsonl(path: Path):
    text = path.read_text(
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



def formula_key(formula):
    """
    Compare molecular formulas by elemental composition,
    not by literal string order.

    Examples:
      C19H18ClN3O4 == C19H18N3O4Cl
      C24H26FNO4   == C24H26NO4F
    """

    if formula is None:
        return None

    text = re.sub(
        r"\s+",
        "",
        str(formula)
    )

    if not text:
        return None

    parts = re.findall(
        r"([A-Z][a-z]?)(\d*)",
        text
    )

    if "".join(
        elem + count
        for elem, count in parts
    ) != text:
        return None

    counts = {}

    for elem, count in parts:
        counts[elem] = (
            counts.get(elem, 0)
            + int(count or "1")
        )

    return tuple(
        sorted(counts.items())
    )


def key2d(smiles):

    if not smiles:
        return None

    mol = Chem.MolFromSmiles(
        str(smiles)
    )

    if mol is None:
        return None

    try:
        return (
            Chem.MolToInchiKey(
                mol
            ).split("-")[0]
        )
    except Exception:
        return Chem.MolToSmiles(
            mol,
            canonical=True,
            isomericSmiles=False,
        )


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--exp1",
        required=True
    )

    ap.add_argument(
        "--massspecgym",
        required=True
    )

    ap.add_argument(
        "--global-pool",
        required=True
    )

    ap.add_argument(
        "--output",
        required=True
    )

    ap.add_argument(
        "--limit",
        type=int,
        default=0
    )

    args = ap.parse_args()

    exp1_rows = read_json_or_jsonl(
        Path(args.exp1)
    )

    dataset_rows = read_json_or_jsonl(
        Path(args.massspecgym)
    )

    pool_rows = read_json_or_jsonl(
        Path(args.global_pool)
    )

    if args.limit > 0:
        exp1_rows = exp1_rows[
            :args.limit
        ]

    exp1_by_id = {
        str(x["identifier"]): x
        for x in exp1_rows
    }

    dataset_by_id = {
        str(x["identifier"]): x
        for x in dataset_rows
    }

    pool_by_id = {
        str(x["sample_id"]): x
        for x in pool_rows
    }

    ids = [
        str(x["identifier"])
        for x in exp1_rows
        if str(x["identifier"])
        in pool_by_id
    ]

    formula_top1 = 0
    formula_top3 = 0
    formula_top5 = 0

    true_formula_branch_generated = 0

    structure_branch_top1 = 0
    structure_branch_top10 = 0

    structure_global_any = 0

    true_structure_formula_rank = Counter()
    true_structure_branch_rank = Counter()

    per_sample = []

    for identifier in ids:

        exp1 = exp1_by_id[
            identifier
        ]

        dataset = dataset_by_id[
            identifier
        ]

        pool = pool_by_id[
            identifier
        ]

        true_formula = str(
            dataset["molecule"]["formula"]
        )

        true_smiles = str(
            dataset["molecule"]["smiles"]
        )

        truth_key = key2d(
            true_smiles
        )

        shortlist = exp1.get(
            "shortlist"
        ) or []

        predicted_formulas = [
            str(x.get("formula") or "")
            for x in shortlist[:5]
        ]

        true_formula_key = formula_key(
            true_formula
        )

        predicted_formula_keys = [
            formula_key(x)
            for x in predicted_formulas
        ]

        if (
            predicted_formula_keys
            and true_formula_key
            == predicted_formula_keys[0]
        ):
            formula_top1 += 1

        if (
            true_formula_key
            in predicted_formula_keys[:3]
        ):
            formula_top3 += 1

        if (
            true_formula_key
            in predicted_formula_keys[:5]
        ):
            formula_top5 += 1

        candidates = pool.get(
            "candidates"
        ) or []

        true_formula_candidates = [
            c
            for c in candidates
            if formula_key(
                c.get("formula")
            ) == true_formula_key
        ]

        if true_formula_candidates:
            true_formula_branch_generated += 1

        exact_hits = []

        for c in candidates:

            candidate_key = (
                c.get("inchikey2d")
                or key2d(
                    c.get("smiles")
                )
            )

            if (
                candidate_key
                and candidate_key
                == truth_key
            ):
                exact_hits.append(c)

        if exact_hits:
            structure_global_any += 1

        branch_exact_hits = [
            c
            for c in true_formula_candidates
            if (
                (
                    c.get("inchikey2d")
                    or key2d(
                        c.get("smiles")
                    )
                )
                == truth_key
            )
        ]

        branch_best_rank = None
        hit_formula_rank = None

        if branch_exact_hits:

            branch_best_rank = min(
                int(
                    c[
                        "branch_route_rank"
                    ]
                )
                for c in branch_exact_hits
            )

            hit_formula_rank = min(
                int(
                    c["formula_rank"]
                )
                for c in branch_exact_hits
            )

            if branch_best_rank == 1:
                structure_branch_top1 += 1

            if branch_best_rank <= 10:
                structure_branch_top10 += 1

            true_structure_formula_rank[
                hit_formula_rank
            ] += 1

            true_structure_branch_rank[
                branch_best_rank
            ] += 1

        per_sample.append(
            {
                "identifier":
                    identifier,

                "true_formula":
                    true_formula,

                "formula_top1_hit":
                    (
                        true_formula_key
                        in predicted_formula_keys[:1]
                    ),

                "formula_top3_hit":
                    (
                        true_formula_key
                        in predicted_formula_keys[:3]
                    ),

                "formula_top5_hit":
                    (
                        true_formula_key
                        in predicted_formula_keys[:5]
                    ),

                "true_formula_predicted_rank":
                    (
                        predicted_formula_keys.index(
                            true_formula_key
                        ) + 1
                        if true_formula_key
                        in predicted_formula_keys
                        else None
                    ),

                "true_formula_branch_has_candidates":
                    bool(
                        true_formula_candidates
                    ),

                "true_structure_in_global_pool":
                    bool(
                        exact_hits
                    ),

                "true_structure_formula_rank":
                    hit_formula_rank,

                "true_structure_branch_rank":
                    branch_best_rank,
            }
        )

    n = len(ids)

    metrics = {
        "num_samples":
            n,

        "formula_top1_accuracy":
            formula_top1 / n
            if n else 0.0,

        "formula_top3_accuracy":
            formula_top3 / n
            if n else 0.0,

        "formula_top5_accuracy":
            formula_top5 / n
            if n else 0.0,

        "true_formula_branch_generation_coverage":
            true_formula_branch_generated / n
            if n else 0.0,

        # Conditional structure-generation diagnostics
        # before cross-formula final ranking.
        "structure_exact_branch_top1_over_all_samples":
            structure_branch_top1 / n
            if n else 0.0,

        "structure_exact_branch_top10_over_all_samples":
            structure_branch_top10 / n
            if n else 0.0,

        "true_structure_global_pool_recall":
            structure_global_any / n
            if n else 0.0,

        "true_structure_formula_rank_distribution":
            {
                str(k): v
                for k, v in sorted(
                    true_structure_formula_rank.items()
                )
            },

        "true_structure_branch_rank_distribution":
            {
                str(k): v
                for k, v in sorted(
                    true_structure_branch_rank.items()
                )
            },

        "important_note":
            (
                "No cross-formula final ranking has been applied. "
                "These are candidate-pool and branch-level diagnostics, "
                "not final Top-1/Top-10 structure metrics."
            ),

        "per_sample":
            per_sample,
    }

    output = Path(
        args.output
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output.write_text(
        json.dumps(
            metrics,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print(
        json.dumps(
            {
                k: v
                for k, v in metrics.items()
                if k != "per_sample"
            },
            ensure_ascii=False,
            indent=2
        )
    )

    print()
    print("===== PER SAMPLE =====")

    for row in per_sample:

        print(
            row["identifier"],
            "| Formula rank =",
            row[
                "true_formula_predicted_rank"
            ],
            "| true formula branch =",
            row[
                "true_formula_branch_has_candidates"
            ],
            "| truth in pool =",
            row[
                "true_structure_in_global_pool"
            ],
            "| truth branch rank =",
            row[
                "true_structure_branch_rank"
            ],
        )


if __name__ == "__main__":
    main()
