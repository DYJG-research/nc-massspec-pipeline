#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors


SYSTEM_PROMPT = """You are an LC-MS/MS pairwise molecular-structure verifier.

You are given one tandem mass spectrum and exactly two candidate molecular structures.
Both candidates have already passed upstream chemical-validity checks.

Your task is to decide which candidate is more consistent with the observed MS/MS evidence.

Important rules:
1. Select exactly one candidate: A or B.
2. Do not generate a new molecular formula.
3. Do not generate, modify, or rewrite any SMILES.
4. Do not invent peaks, neutral losses, fragments, or reaction mechanisms.
5. Upstream machine-learning ranks and ranking scores are intentionally hidden.
6. Base your comparison on the observed spectrum, molecular formula, adduct context,
   molecular structure, and deterministic structural properties supplied below.
7. If the evidence is subtle, still make the best pairwise choice from the supplied candidates.
8. Output only:
<selection>A</selection>
or
<selection>B</selection>
"""


SMARTS = {
    "carbonyl": "[CX3]=[OX1]",
    "hydroxyl": "[OX2H]",
    "amide": "[NX3][CX3](=[OX1])",
    "ester": "[CX3](=[OX1])[OX2][#6]",
    "carboxyl": "[CX3](=[OX1])[OX2H1]",
}

SMARTS_MOLS = {
    name: Chem.MolFromSmarts(smarts)
    for name, smarts in SMARTS.items()
}


def read_records(path: Path):
    text = path.read_text(encoding="utf-8").strip()

    if not text:
        return []

    if text.startswith("[") or text.startswith("{"):
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            obj = None

        if isinstance(obj, list):
            return obj

        if isinstance(obj, dict):
            for key in ("records", "data", "rows", "results"):
                if isinstance(obj.get(key), list):
                    return obj[key]

            rows = []
            for key, value in obj.items():
                if isinstance(value, dict):
                    row = dict(value)
                    row.setdefault("identifier", key)
                    rows.append(row)

            if rows:
                return rows

    rows = []

    for line in text.splitlines():
        if line.strip():
            rows.append(json.loads(line))

    return rows


def safe_float(x, default=None):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return default

    if not math.isfinite(x):
        return default

    return x


def fmt(x, digits=4):
    x = safe_float(x)

    if x is None:
        return "NA"

    return f"{x:.{digits}f}"


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

    output = []
    seen = set()

    for row in ordered:
        smiles = str(row.get("smiles") or "")

        if not smiles or smiles in seen:
            continue

        seen.add(smiles)
        output.append(row)

    return output


def top_peaks(mass_row, max_peaks=30):
    spectrum = mass_row.get("spectrum") or {}
    peaks = spectrum.get("peaks") or []

    parsed = []

    for p in peaks:
        if not isinstance(p, dict):
            continue

        mz = safe_float(p.get("mz"))
        intensity = safe_float(p.get("intensity"))

        if mz is None or intensity is None:
            continue

        parsed.append((mz, intensity))

    if not parsed:
        return []

    selected = sorted(
        parsed,
        key=lambda x: x[1],
        reverse=True,
    )[:max_peaks]

    max_i = max(x[1] for x in selected) or 1.0

    selected = [
        (mz, intensity / max_i * 100.0)
        for mz, intensity in selected
    ]

    selected.sort(key=lambda x: x[0])

    return selected


def spectrum_text(
    precursor_mz,
    polarity,
    peaks,
):
    peak_string = ", ".join(
        f"{mz:.4f}:{inten:.1f}"
        for mz, inten in peaks
    )

    # Purely observed precursor-to-peak mass differences.
    losses = []

    for mz, inten in peaks:
        loss = precursor_mz - mz

        if 2.0 <= loss <= 250.0:
            losses.append(
                (loss, inten)
            )

    losses = sorted(
        losses,
        key=lambda x: x[1],
        reverse=True,
    )[:15]

    loss_string = ", ".join(
        f"{loss:.4f}"
        for loss, _ in losses
    )

    if not loss_string:
        loss_string = "none"

    return (
        f"Precursor m/z: {precursor_mz:.6f}\n"
        f"Ionization polarity: {polarity}\n\n"
        "Observed MS/MS peaks are shown as "
        "m/z:relative_intensity:\n"
        f"{peak_string}\n\n"
        "Observed precursor-to-fragment mass differences "
        "(unassigned, Da):\n"
        f"{loss_string}"
    )


def substructure_count(mol, name):
    query = SMARTS_MOLS[name]

    if query is None:
        return 0

    return len(
        mol.GetSubstructMatches(query)
    )


def candidate_visible_block(
    score_row,
    pool_row,
):
    smiles = str(
        score_row.get("smiles")
        or pool_row.get("smiles")
        or ""
    )

    formula = str(
        score_row.get("formula")
        or pool_row.get("formula")
        or ""
    )

    adduct = str(
        pool_row.get("adduct")
        or "unknown"
    )

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        descriptors = {
            "heavy_atoms": "NA",
            "hetero_atoms": "NA",
            "ring_count": "NA",
            "aromatic_rings": "NA",
            "rotatable_bonds": "NA",
            "hbd": "NA",
            "hba": "NA",
            "tpsa": "NA",
            "fraction_csp3": "NA",
            "formal_charge": "NA",
            "carbonyl": "NA",
            "hydroxyl": "NA",
            "amide": "NA",
            "ester": "NA",
            "carboxyl": "NA",
        }

    else:
        descriptors = {
            "heavy_atoms":
                mol.GetNumHeavyAtoms(),

            "hetero_atoms":
                rdMolDescriptors.CalcNumHeteroatoms(mol),

            "ring_count":
                rdMolDescriptors.CalcNumRings(mol),

            "aromatic_rings":
                rdMolDescriptors.CalcNumAromaticRings(mol),

            "rotatable_bonds":
                Lipinski.NumRotatableBonds(mol),

            "hbd":
                Lipinski.NumHDonors(mol),

            "hba":
                Lipinski.NumHAcceptors(mol),

            "tpsa":
                f"{rdMolDescriptors.CalcTPSA(mol):.2f}",

            "fraction_csp3":
                f"{rdMolDescriptors.CalcFractionCSP3(mol):.3f}",

            "formal_charge":
                Chem.GetFormalCharge(mol),

            "carbonyl":
                substructure_count(mol, "carbonyl"),

            "hydroxyl":
                substructure_count(mol, "hydroxyl"),

            "amide":
                substructure_count(mol, "amide"),

            "ester":
                substructure_count(mol, "ester"),

            "carboxyl":
                substructure_count(mol, "carboxyl"),
        }

    return "\n".join(
        [
            f"Formula: {formula}",
            f"Adduct: {adduct}",
            f"SMILES: {smiles}",
            f"Absolute formula ppm error: "
            f"{fmt(pool_row.get('formula_ppm_error'))}",
            f"DBE: {fmt(pool_row.get('formula_dbe'))}",
            "",
            "Deterministic structure descriptors:",
            f"Heavy atoms: {descriptors['heavy_atoms']}",
            f"Hetero atoms: {descriptors['hetero_atoms']}",
            f"Ring count: {descriptors['ring_count']}",
            f"Aromatic rings: {descriptors['aromatic_rings']}",
            f"Rotatable bonds: {descriptors['rotatable_bonds']}",
            f"H-bond donors: {descriptors['hbd']}",
            f"H-bond acceptors: {descriptors['hba']}",
            f"TPSA: {descriptors['tpsa']}",
            f"Fraction Csp3: {descriptors['fraction_csp3']}",
            f"Formal charge: {descriptors['formal_charge']}",
            f"Carbonyl groups: {descriptors['carbonyl']}",
            f"Hydroxyl groups: {descriptors['hydroxyl']}",
            f"Amide groups: {descriptors['amide']}",
            f"Ester groups: {descriptors['ester']}",
            f"Carboxyl groups: {descriptors['carboxyl']}",
        ]
    )


def make_pair_prompt(
    spectrum_block,
    block_a,
    block_b,
):
    return (
        f"{spectrum_block}\n\n"
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


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--scores", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--massspecgym", required=True)
    ap.add_argument("--exp1", required=True)
    ap.add_argument("--output-dir", required=True)

    ap.add_argument(
        "--max-peaks",
        type=int,
        default=30,
    )

    ap.add_argument(
        "--pair-top-k",
        type=int,
        default=10,
    )

    ap.add_argument(
        "--keep-hard-negatives",
        type=int,
        default=2,
    )

    args = ap.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    score_rows = read_records(
        Path(args.scores)
    )

    pool_rows = read_records(
        Path(args.pool)
    )

    mass_rows = read_records(
        Path(args.massspecgym)
    )

    exp1_rows = read_records(
        Path(args.exp1)
    )

    scores_by_sample = defaultdict(list)

    for row in score_rows:
        scores_by_sample[
            str(row["sample_id"])
        ].append(row)

    pool_by_key = {}

    for sample in pool_rows:
        sid = str(sample["sample_id"])

        for candidate in (
            sample.get("candidates")
            or []
        ):
            pool_by_key[
                (
                    sid,
                    str(candidate["candidate_id"]),
                )
            ] = candidate

    mass_by_id = {
        str(row["identifier"]): row
        for row in mass_rows
    }

    exp1_by_id = {
        str(row["identifier"]): row
        for row in exp1_rows
    }

    supervised_path = (
        outdir
        / "pairwise_supervised_full.jsonl"
    )

    context_path = (
        outdir
        / "pairwise_context.jsonl"
    )

    stats = Counter()
    truth_rank_dist = Counter()
    fold_supervised_base = Counter()

    with (
        supervised_path.open(
            "w",
            encoding="utf-8",
        ) as sup_f,
        context_path.open(
            "w",
            encoding="utf-8",
        ) as ctx_f,
    ):
        for sid in sorted(
            scores_by_sample.keys()
        ):
            ranked = rank_candidates(
                scores_by_sample[sid]
            )

            if not ranked:
                continue

            topk = ranked[
                :args.pair_top_k
            ]

            if not topk:
                continue

            mass_row = mass_by_id.get(sid)
            exp1_row = exp1_by_id.get(sid)

            if (
                mass_row is None
                or exp1_row is None
            ):
                raise RuntimeError(
                    f"Missing metadata for {sid}"
                )

            precursor_mz = float(
                exp1_row["precursor_mz"]
            )

            polarity = str(
                exp1_row["polarity"]
            )

            peaks = top_peaks(
                mass_row,
                args.max_peaks,
            )

            spec_block = spectrum_text(
                precursor_mz,
                polarity,
                peaks,
            )

            candidate_contexts = []

            truth_rank = None
            truth_id = None

            for rank, row in enumerate(
                topk,
                1,
            ):
                cid = str(
                    row["candidate_id"]
                )

                pool_row = pool_by_key.get(
                    (sid, cid)
                )

                if pool_row is None:
                    raise RuntimeError(
                        f"Missing pool candidate: "
                        f"{sid}/{cid}"
                    )

                visible = candidate_visible_block(
                    row,
                    pool_row,
                )

                candidate_contexts.append(
                    {
                        "candidate_id":
                            cid,

                        # Orchestration only.
                        # Never shown to the LLM.
                        "xgb_rank":
                            rank,

                        "visible_block":
                            visible,
                    }
                )

                if int(
                    row.get("label", 0)
                ) == 1:
                    truth_rank = rank
                    truth_id = cid

            fold = int(
                topk[0]["fold"]
            )

            margin = None

            if len(topk) >= 2:
                margin = (
                    float(
                        topk[0]["oof_score"]
                    )
                    -
                    float(
                        topk[1]["oof_score"]
                    )
                )

            # No GT in inference context.
            context_row = {
                "sample_id":
                    sid,

                "fold":
                    fold,

                "xgb_margin_top1_top2":
                    margin,

                "system_prompt":
                    SYSTEM_PROMPT,

                "spectrum_block":
                    spec_block,

                "candidates":
                    candidate_contexts,
            }

            ctx_f.write(
                json.dumps(
                    context_row,
                    ensure_ascii=False,
                )
                + "\n"
            )

            stats[
                "context_samples"
            ] += 1

            if truth_id is None:
                stats[
                    "no_truth_in_pair_topk"
                ] += 1
                continue

            stats[
                "supervised_base_samples"
            ] += 1

            fold_supervised_base[
                str(fold)
            ] += 1

            truth_rank_dist[
                str(truth_rank)
            ] += 1

            truth_item = next(
                x
                for x in candidate_contexts
                if x["candidate_id"]
                == truth_id
            )

            negatives = [
                x
                for x in candidate_contexts
                if x["candidate_id"]
                != truth_id
            ]

            # XGB Top1 already correct:
            # only learn against the hardest nearby negatives.
            if truth_rank == 1:
                negatives = negatives[
                    :args.keep_hard_negatives
                ]

                pair_type = "keep"

            else:
                # Correction case:
                # compare truth against every other Top-K candidate.
                pair_type = "correction"

            for neg in negatives:

                pair_key = (
                    f"{sid}::"
                    f"{truth_id}::"
                    f"{neg['candidate_id']}"
                )

                # Orientation 1:
                # truth = A
                user_1 = make_pair_prompt(
                    spec_block,
                    truth_item[
                        "visible_block"
                    ],
                    neg[
                        "visible_block"
                    ],
                )

                row_1 = {
                    "sample_id":
                        sid,

                    "fold":
                        fold,

                    "pair_key":
                        pair_key,

                    "pair_type":
                        pair_type,

                    "truth_xgb_rank":
                        truth_rank,

                    "truth_candidate_id":
                        truth_id,

                    "negative_candidate_id":
                        neg["candidate_id"],

                    "orientation":
                        "truth_A",

                    "messages": [
                        {
                            "role": "system",
                            "content":
                                SYSTEM_PROMPT,
                        },
                        {
                            "role": "user",
                            "content":
                                user_1,
                        },
                        {
                            "role": "assistant",
                            "content":
                                "<selection>A</selection>",
                        },
                    ],
                }

                sup_f.write(
                    json.dumps(
                        row_1,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                # Orientation 2:
                # truth = B
                user_2 = make_pair_prompt(
                    spec_block,
                    neg[
                        "visible_block"
                    ],
                    truth_item[
                        "visible_block"
                    ],
                )

                row_2 = dict(
                    row_1
                )

                row_2[
                    "orientation"
                ] = "truth_B"

                row_2[
                    "messages"
                ] = [
                    {
                        "role": "system",
                        "content":
                            SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content":
                            user_2,
                    },
                    {
                        "role": "assistant",
                        "content":
                            "<selection>B</selection>",
                    },
                ]

                sup_f.write(
                    json.dumps(
                        row_2,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                stats[
                    f"{pair_type}_pair_rows"
                ] += 2

                stats[
                    "supervised_pair_rows"
                ] += 2

    summary = {
        **dict(stats),

        "pair_top_k":
            args.pair_top_k,

        "keep_hard_negatives":
            args.keep_hard_negatives,

        "truth_rank_distribution":
            dict(
                sorted(
                    truth_rank_dist.items(),
                    key=lambda x:
                        int(x[0])
                )
            ),

        "fold_supervised_base":
            dict(
                sorted(
                    fold_supervised_base.items()
                )
            ),

        "llm_visible_xgb_rank":
            False,

        "llm_visible_xgb_score":
            False,

        "llm_visible_csi_score":
            False,

        "assistant_target":
            "<selection>A</selection> or "
            "<selection>B</selection>",
    }

    (
        outdir
        / "pairwise_dataset_stats.json"
    ).write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
