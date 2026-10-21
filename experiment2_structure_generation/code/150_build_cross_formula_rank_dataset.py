#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


SOURCE_TYPES = [
    "own_top1_lock",
    "own_msnovelist",
    "same_formula_batch_transfer",
    "formula_delta_dehydration",
    "formula_delta_deacetylation",
    "formula_delta_tri_hydroxylation",
    "formula_delta_double_reduction",
]

ADDUCT_TYPES = [
    "[M+H]+",
    "[M+Na]+",
    "[M+NH4]+",
    "[M+K]+",
    "[M-H]-",
    "[M+Cl]-",
    "[M+FA-H]-",
    "[M+Ac-H]-",
]


def load_rows(path: Path):
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


def safe_float(x, default=0.0):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return default

    if not math.isfinite(x):
        return default

    return x


def is_valid_number(x):
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def mean_std(values):
    vals = [
        float(x)
        for x in values
        if is_valid_number(x)
    ]

    if not vals:
        return 0.0, 1.0

    mean = sum(vals) / len(vals)

    var = sum(
        (x - mean) ** 2
        for x in vals
    ) / len(vals)

    std = math.sqrt(var)

    if std < 1e-8:
        std = 1.0

    return mean, std


def median(values):
    vals = sorted(
        float(x)
        for x in values
        if is_valid_number(x)
    )

    if not vals:
        return 0.0

    n = len(vals)

    if n % 2:
        return vals[n // 2]

    return (
        vals[n // 2 - 1]
        + vals[n // 2]
    ) / 2.0


def mol2d_key(smiles):
    if not smiles:
        return None

    try:
        mol = Chem.MolFromSmiles(
            str(smiles)
        )
    except Exception:
        return None

    if mol is None:
        return None

    try:
        return (
            Chem.MolToInchiKey(mol)
            .split("-")[0]
        )
    except Exception:
        return None


def make_features(
    row,
    candidate,
    sample_context,
):
    source = str(
        candidate.get(
            "structure_source"
        )
        or "unknown"
    )

    adduct = str(
        candidate.get("adduct")
        or ""
    )

    formula_rank = int(
        candidate.get(
            "formula_rank",
            999
        )
    )

    branch_rank = int(
        candidate.get(
            "branch_route_rank",
            999
        )
    )

    formula_score = safe_float(
        candidate.get(
            "formula_score"
        )
    )

    ppm = safe_float(
        candidate.get(
            "formula_ppm_error"
        )
    )

    dbe = safe_float(
        candidate.get(
            "formula_dbe"
        )
    )

    source_rank_raw = candidate.get(
        "structure_source_rank"
    )

    source_rank_missing = int(
        not is_valid_number(
            source_rank_raw
        )
    )

    source_rank = (
        safe_float(
            source_rank_raw
        )
        if not source_rank_missing
        else 0.0
    )

    structure_raw = candidate.get(
        "structure_source_score"
    )

    structure_missing = int(
        not is_valid_number(
            structure_raw
        )
    )

    structure_score = (
        safe_float(
            structure_raw
        )
        if not structure_missing
        else sample_context[
            "structure_score_median"
        ]
    )

    fmean = sample_context[
        "formula_score_mean"
    ]

    fstd = sample_context[
        "formula_score_std"
    ]

    formula_score_z = (
        formula_score - fmean
    ) / fstd

    formula_score_margin_best = (
        formula_score
        - sample_context[
            "formula_score_best"
        ]
    )

    smean = sample_context[
        "structure_score_mean"
    ]

    sstd = sample_context[
        "structure_score_std"
    ]

    structure_score_sample_z = (
        structure_score - smean
    ) / sstd

    branch_context = (
        sample_context[
            "branch_structure_stats"
        ].get(
            formula_rank,
            {
                "mean": smean,
                "std": sstd,
                "best": structure_score,
                "count": 0,
            }
        )
    )

    structure_score_branch_z = (
        structure_score
        - branch_context["mean"]
    ) / branch_context["std"]

    structure_score_margin_branch_best = (
        structure_score
        - branch_context["best"]
    )

    sirius = (
        candidate.get(
            "sirius_scores"
        )
        or {}
    )

    csi = safe_float(
        sirius.get(
            "CSI:FingerIDScore"
        ),
        structure_score,
    )

    model_score = safe_float(
        sirius.get(
            "ModelScore"
        )
    )

    sirius_norm = safe_float(
        sirius.get(
            "SiriusScoreNormalized"
        )
    )

    sirius_score = safe_float(
        sirius.get(
            "SiriusScore"
        )
    )

    spectral_similarity = safe_float(
        candidate.get(
            "spectral_similarity"
        )
    )

    transformation_raw = candidate.get(
        "transformation_score"
    )

    transformation_missing = int(
        not is_valid_number(
            transformation_raw
        )
    )

    transformation_score = (
        safe_float(
            transformation_raw
        )
        if not transformation_missing
        else 0.0
    )

    site_raw = candidate.get(
        "site_score"
    )

    site_missing = int(
        not is_valid_number(
            site_raw
        )
    )

    site_score = (
        safe_float(site_raw)
        if not site_missing
        else 0.0
    )

    donor_rank = int(
        candidate.get(
            "donor_formula_rank",
            formula_rank
        )
    )

    donor_formula_score = safe_float(
        candidate.get(
            "donor_formula_score"
        ),
        formula_score,
    )

    donor_sample_id = str(
        candidate.get(
            "donor_sample_id"
        )
        or ""
    )

    donor_branch_id = str(
        candidate.get(
            "donor_branch_id"
        )
        or ""
    )

    sample_id = str(
        row["sample_id"]
    )

    branch_id = str(
        candidate.get(
            "branch_id"
        )
        or ""
    )

    features = {
        # Formula evidence
        "formula_rank":
            float(formula_rank),

        "formula_rank_recip":
            1.0 / max(
                formula_rank,
                1
            ),

        "formula_score":
            formula_score,

        "formula_score_z":
            formula_score_z,

        "formula_score_margin_best":
            formula_score_margin_best,

        "abs_formula_ppm_error":
            abs(ppm),

        "formula_dbe":
            dbe,

        # Branch evidence
        "branch_route_rank":
            float(branch_rank),

        "branch_rank_recip":
            1.0 / max(
                branch_rank,
                1
            ),

        "branch_candidate_count":
            float(
                branch_context[
                    "count"
                ]
            ),

        # Structure evidence
        "structure_source_rank":
            source_rank,

        "structure_source_rank_missing":
            float(
                source_rank_missing
            ),

        "structure_source_rank_recip":
            (
                1.0 / max(
                    source_rank,
                    1.0
                )
                if not source_rank_missing
                else 0.0
            ),

        "structure_source_score":
            structure_score,

        "structure_score_missing":
            float(
                structure_missing
            ),

        "structure_score_sample_z":
            structure_score_sample_z,

        "structure_score_branch_z":
            structure_score_branch_z,

        "structure_score_margin_branch_best":
            structure_score_margin_branch_best,

        # SIRIUS evidence
        "csi_fingerid_score":
            csi,

        "model_score":
            model_score,

        "sirius_score_normalized":
            sirius_norm,

        "sirius_score":
            sirius_score,

        # Formula-Lattice evidence
        "spectral_similarity":
            spectral_similarity,

        "transformation_score":
            transformation_score,

        "transformation_score_missing":
            float(
                transformation_missing
            ),

        "site_score":
            site_score,

        "site_score_missing":
            float(
                site_missing
            ),

        # Donor evidence
        "donor_formula_rank":
            float(donor_rank),

        "donor_formula_rank_recip":
            1.0 / max(
                donor_rank,
                1
            ),

        "donor_formula_score":
            donor_formula_score,

        "donor_formula_score_delta":
            donor_formula_score
            - formula_score,

        "same_sample_donor":
            float(
                donor_sample_id
                == sample_id
            ),

        "same_branch_donor":
            float(
                donor_branch_id
                == branch_id
            ),

        "same_formula_rank_donor":
            float(
                donor_rank
                == formula_rank
            ),

        # Sample context
        "num_formula_branches":
            float(
                row.get(
                    "num_formula_branches",
                    0
                )
            ),

        "num_nonempty_formula_branches":
            float(
                row.get(
                    "num_nonempty_formula_branches",
                    0
                )
            ),

        "num_sample_candidates":
            float(
                row.get(
                    "num_candidates",
                    0
                )
            ),

        # Provenance summary
        "is_own_candidate":
            float(
                source in {
                    "own_top1_lock",
                    "own_msnovelist",
                }
            ),

        "is_same_formula_transfer":
            float(
                source
                == "same_formula_batch_transfer"
            ),

        "is_formula_delta":
            float(
                source.startswith(
                    "formula_delta_"
                )
            ),
    }

    for s in SOURCE_TYPES:
        features[
            f"source__{s}"
        ] = float(
            source == s
        )

    features[
        "source__other"
    ] = float(
        source
        not in SOURCE_TYPES
    )

    for a in ADDUCT_TYPES:
        features[
            f"adduct__{a}"
        ] = float(
            adduct == a
        )

    features[
        "adduct__other"
    ] = float(
        adduct
        not in ADDUCT_TYPES
    )

    return features


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--pool",
        required=True
    )

    ap.add_argument(
        "--massspecgym",
        required=True
    )

    ap.add_argument(
        "--output-dir",
        required=True
    )

    args = ap.parse_args()

    pool_path = Path(
        args.pool
    )

    mass_path = Path(
        args.massspecgym
    )

    outdir = Path(
        args.output_dir
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True
    )

    mass_rows = load_rows(
        mass_path
    )

    truth = {}

    for row in mass_rows:
        sid = str(
            row["identifier"]
        )

        smiles = str(
            row["molecule"]["smiles"]
        )

        truth[sid] = {
            "smiles":
                smiles,

            "key2d":
                mol2d_key(
                    smiles
                ),
        }

    feature_path = (
        outdir
        / "rank_features.jsonl"
    )

    label_path = (
        outdir
        / "rank_labels.jsonl"
    )

    manifest_path = (
        outdir
        / "sample_manifest.jsonl"
    )

    feature_f = feature_path.open(
        "w",
        encoding="utf-8"
    )

    label_f = label_path.open(
        "w",
        encoding="utf-8"
    )

    manifest_f = manifest_path.open(
        "w",
        encoding="utf-8"
    )

    stats = Counter()

    positive_samples = set()

    feature_names = None

    try:
        with pool_path.open(
            encoding="utf-8"
        ) as f:

            for line in f:
                if not line.strip():
                    continue

                row = json.loads(
                    line
                )

                sid = str(
                    row["sample_id"]
                )

                if sid not in truth:
                    raise RuntimeError(
                        f"Missing truth: {sid}"
                    )

                truth_key = (
                    truth[sid][
                        "key2d"
                    ]
                )

                split_group = (
                    truth_key
                    if truth_key
                    else
                    f"INVALID::{sid}"
                )

                candidates = (
                    row.get(
                        "candidates"
                    )
                    or []
                )

                # Unique formula-level scores.
                formula_score_by_rank = {}

                for c in candidates:
                    rank = int(
                        c.get(
                            "formula_rank",
                            999
                        )
                    )

                    formula_score_by_rank[
                        rank
                    ] = safe_float(
                        c.get(
                            "formula_score"
                        )
                    )

                formula_scores = list(
                    formula_score_by_rank.values()
                )

                fmean, fstd = mean_std(
                    formula_scores
                )

                fbest = (
                    max(formula_scores)
                    if formula_scores
                    else 0.0
                )

                structure_scores = [
                    c.get(
                        "structure_source_score"
                    )
                    for c in candidates
                    if is_valid_number(
                        c.get(
                            "structure_source_score"
                        )
                    )
                ]

                smean, sstd = mean_std(
                    structure_scores
                )

                smedian = median(
                    structure_scores
                )

                branch_scores = defaultdict(
                    list
                )

                branch_counts = Counter()

                for c in candidates:
                    rank = int(
                        c.get(
                            "formula_rank",
                            999
                        )
                    )

                    branch_counts[
                        rank
                    ] += 1

                    score = c.get(
                        "structure_source_score"
                    )

                    if is_valid_number(
                        score
                    ):
                        branch_scores[
                            rank
                        ].append(
                            float(score)
                        )

                branch_structure_stats = {}

                for rank in branch_counts:
                    vals = branch_scores.get(
                        rank,
                        []
                    )

                    bmean, bstd = mean_std(
                        vals
                    )

                    bbest = (
                        max(vals)
                        if vals
                        else smedian
                    )

                    branch_structure_stats[
                        rank
                    ] = {
                        "mean":
                            bmean,

                        "std":
                            bstd,

                        "best":
                            bbest,

                        "count":
                            branch_counts[
                                rank
                            ],
                    }

                context = {
                    "formula_score_mean":
                        fmean,

                    "formula_score_std":
                        fstd,

                    "formula_score_best":
                        fbest,

                    "structure_score_mean":
                        smean,

                    "structure_score_std":
                        sstd,

                    "structure_score_median":
                        smedian,

                    "branch_structure_stats":
                        branch_structure_stats,
                }

                num_positive = 0

                for c in candidates:
                    candidate_id = str(
                        c["candidate_id"]
                    )

                    smiles = str(
                        c["smiles"]
                    )

                    candidate_key = (
                        mol2d_key(
                            smiles
                        )
                    )

                    label = int(
                        truth_key is not None
                        and candidate_key
                        == truth_key
                    )

                    num_positive += label

                    features = make_features(
                        row,
                        c,
                        context,
                    )

                    if feature_names is None:
                        feature_names = list(
                            features.keys()
                        )
                    else:
                        if set(
                            features.keys()
                        ) != set(
                            feature_names
                        ):
                            raise RuntimeError(
                                "Feature schema mismatch"
                            )

                    feature_f.write(
                        json.dumps(
                            {
                                "sample_id":
                                    sid,

                                "candidate_id":
                                    candidate_id,

                                "smiles":
                                    smiles,

                                "formula":
                                    c.get(
                                        "formula"
                                    ),

                                "formula_rank":
                                    c.get(
                                        "formula_rank"
                                    ),

                                "branch_route_rank":
                                    c.get(
                                        "branch_route_rank"
                                    ),

                                "features":
                                    features,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                    label_f.write(
                        json.dumps(
                            {
                                "sample_id":
                                    sid,

                                "candidate_id":
                                    candidate_id,

                                "label":
                                    label,

                                "split_group":
                                    split_group,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                    stats[
                        "candidate_rows"
                    ] += 1

                    stats[
                        "positive_candidate_rows"
                    ] += label

                if num_positive > 0:
                    positive_samples.add(
                        sid
                    )

                manifest_f.write(
                    json.dumps(
                        {
                            "sample_id":
                                sid,

                            "split_group":
                                split_group,

                            "num_candidates":
                                len(candidates),

                            "num_positive_candidates":
                                num_positive,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                stats[
                    "sample_rows"
                ] += 1

                if candidates:
                    stats[
                        "nonempty_samples"
                    ] += 1

    finally:
        feature_f.close()
        label_f.close()
        manifest_f.close()

    stats[
        "positive_samples"
    ] = len(
        positive_samples
    )

    output_stats = {
        **dict(stats),

        "num_features":
            len(
                feature_names
                or []
            ),

        "feature_names":
            feature_names
            or [],

        "truth_fields_in_feature_file":
            [],

        "label_definition":
            (
                "2D InChIKey first block "
                "exact match"
            ),

        "split_group_definition":
            (
                "reference 2D InChIKey, "
                "used only for OOF grouping"
            ),
    }

    (
        outdir
        / "rank_dataset_stats.json"
    ).write_text(
        json.dumps(
            output_stats,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8"
    )

    print(
        json.dumps(
            output_stats,
            ensure_ascii=False,
            indent=2,
        )
    )

    print(
        "[OK]",
        feature_path
    )

    print(
        "[OK]",
        label_path
    )

    print(
        "[OK]",
        manifest_path
    )


if __name__ == "__main__":
    main()
