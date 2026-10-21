#!/usr/bin/env python3

import argparse
import json
from collections import defaultdict
from pathlib import Path

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.error")
RDLogger.DisableLog("rdApp.warning")

from multiformula_common import read_jsonl, write_jsonl


def canonical_smiles(smiles):

    mol = Chem.MolFromSmiles(str(smiles))

    if mol is None:
        return None

    if len(Chem.GetMolFrags(mol)) != 1:
        return None

    return Chem.MolToSmiles(
        mol,
        canonical=True,
        isomericSmiles=False,
    )


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--branch-candidates",
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

    by_sample = defaultdict(list)

    branch_rows = 0
    raw_candidates = 0
    invalid_smiles = 0

    branches_by_sample = defaultdict(int)
    nonempty_branches_by_sample = defaultdict(int)

    for _, row in read_jsonl(
        args.branch_candidates
    ):

        branch_rows += 1

        sample_id = str(
            row["sample_id"]
        )

        branch_id = str(
            row["branch_id"]
        )

        formula_rank = int(
            row["formula_rank"]
        )

        formula = str(
            row["formula"]
        )

        formula_score = row.get(
            "formula_score"
        )

        formula_confidence = row.get(
            "formula_confidence"
        )

        formula_ppm_error = row.get(
            "formula_ppm_error"
        )

        formula_dbe = row.get(
            "formula_dbe"
        )

        adduct = row.get(
            "adduct"
        )

        candidates = (
            row.get("candidates")
            or []
        )

        branches_by_sample[
            sample_id
        ] += 1

        if candidates:
            nonempty_branches_by_sample[
                sample_id
            ] += 1

        # IMPORTANT:
        # This enumerate() position is the final ordering produced
        # by SIRIUS or Formula-Lattice for THIS TARGET BRANCH.
        #
        # It must be preserved independently from donor/source ranks.
        for branch_route_rank, candidate in enumerate(
            candidates,
            1,
        ):

            raw_candidates += 1

            smiles = candidate.get(
                "smiles"
            )

            if not smiles:
                continue

            canonical = canonical_smiles(
                smiles
            )

            if canonical is None:
                invalid_smiles += 1
                continue

            sirius_score = (
                candidate.get(
                    "generator_raw"
                )
            )

            structure_source = (
                candidate.get(
                    "prediction_source"
                )
                or candidate.get(
                    "source"
                )
                or "sirius_msnovelist"
            )

            source_rank = candidate.get(
                "rank"
            )

            try:
                source_rank = (
                    int(source_rank)
                    if source_rank is not None
                    else None
                )
            except Exception:
                source_rank = None

            item = {
                "candidate_id":
                    None,

                "smiles":
                    canonical,

                "inchikey2d":
                    candidate.get(
                        "inchikey2d"
                    ),

                "formula":
                    formula,

                "formula_rank":
                    formula_rank,

                "formula_score":
                    formula_score,

                "formula_confidence":
                    formula_confidence,

                "formula_ppm_error":
                    formula_ppm_error,

                "formula_dbe":
                    formula_dbe,

                "adduct":
                    adduct,

                "branch_id":
                    branch_id,

                # THIS is the important rank after branch-level
                # SIRIUS / Formula-Lattice merging.
                "branch_route_rank":
                    branch_route_rank,

                # Original generator/donor rank is retained only
                # as provenance.
                "structure_source_rank":
                    source_rank,

                "structure_source":
                    structure_source,

                "structure_source_score":
                    sirius_score,

                "sirius_scores":
                    candidate.get(
                        "sirius_scores",
                        {}
                    ),

                "spectral_similarity":
                    candidate.get(
                        "spectral_similarity"
                    ),

                "transformation_score":
                    candidate.get(
                        "transformation_score"
                    ),

                "site_score":
                    candidate.get(
                        "site_score"
                    ),

                "donor_branch_id":
                    candidate.get(
                        "donor_branch_id"
                    ),

                "donor_sample_id":
                    candidate.get(
                        "donor_sample_id"
                    ),

                "donor_formula_rank":
                    candidate.get(
                        "donor_formula_rank"
                    ),

                "donor_formula_score":
                    candidate.get(
                        "donor_formula_score"
                    ),

                "donor_formula_confidence":
                    candidate.get(
                        "donor_formula_confidence"
                    ),

                "donor_adduct":
                    candidate.get(
                        "donor_adduct"
                    ),

                "provenance": [
                    {
                        "branch_id":
                            branch_id,

                        "formula":
                            formula,

                        "formula_rank":
                            formula_rank,

                        "branch_route_rank":
                            branch_route_rank,

                        "source":
                            structure_source,

                        "source_rank":
                            source_rank,

                        "source_score":
                            sirius_score,

                        "donor_branch_id":
                            candidate.get(
                                "donor_branch_id"
                            ),

                        "donor_sample_id":
                            candidate.get(
                                "donor_sample_id"
                            ),

                        "donor_formula_rank":
                            candidate.get(
                                "donor_formula_rank"
                            ),

                        "donor_formula_confidence":
                            candidate.get(
                                "donor_formula_confidence"
                            ),

                        "spectral_similarity":
                            candidate.get(
                                "spectral_similarity"
                            ),

                        "transformation_score":
                            candidate.get(
                                "transformation_score"
                            ),
                    }
                ],
            }

            by_sample[
                sample_id
            ].append(
                item
            )

    all_samples = sorted(
        branches_by_sample
    )

    output_rows = []

    unique_total = 0
    duplicate_routes = 0

    for sample_id in all_samples:

        candidates = by_sample.get(
            sample_id,
            []
        )

        dedup = {}

        for candidate in candidates:

            key = candidate["smiles"]

            if key not in dedup:

                dedup[key] = candidate
                continue

            duplicate_routes += 1

            old = dedup[key]

            old[
                "provenance"
            ].extend(
                candidate[
                    "provenance"
                ]
            )

            # Primary route selection:
            #
            # 1. Better predicted Formula rank
            # 2. Better final rank INSIDE that formula branch
            #
            # Do NOT use donor/source rank here.
            old_key = (
                int(
                    old["formula_rank"]
                ),
                int(
                    old["branch_route_rank"]
                ),
            )

            new_key = (
                int(
                    candidate["formula_rank"]
                ),
                int(
                    candidate[
                        "branch_route_rank"
                    ]
                ),
            )

            if new_key < old_key:

                provenance = list(
                    old["provenance"]
                )

                dedup[key] = candidate

                dedup[key][
                    "provenance"
                ] = provenance

        # IMPORTANT:
        #
        # This is only a stable ingestion order.
        #
        # Formula rank is currently the first axis because
        # cross-formula scientific ranking has NOT yet been
        # defined.
        #
        # Within each Formula branch, branch_route_rank exactly
        # preserves Formula-Lattice ordering and Top-1 lock.
        final_candidates = sorted(
            dedup.values(),
            key=lambda x: (
                int(
                    x["formula_rank"]
                ),
                int(
                    x["branch_route_rank"]
                ),
                x["smiles"],
            )
        )

        for idx, candidate in enumerate(
            final_candidates,
            1,
        ):

            candidate[
                "candidate_id"
            ] = f"C{idx:05d}"

            # This is explicitly NOT the final scientific rank.
            candidate[
                "global_ingestion_rank"
            ] = idx

        unique_total += len(
            final_candidates
        )

        output_rows.append(
            {
                "schema_version":
                    "global-formula-structure-pool-v2",

                "sample_id":
                    sample_id,

                "num_formula_branches":
                    branches_by_sample[
                        sample_id
                    ],

                "num_nonempty_formula_branches":
                    nonempty_branches_by_sample[
                        sample_id
                    ],

                "num_candidates":
                    len(
                        final_candidates
                    ),

                "ranking_status":
                    "ingestion_order_only",

                "cross_formula_ranking_applied":
                    False,

                "formula_confidence_used_for_ranking":
                    False,

                "candidates":
                    final_candidates,
            }
        )

    write_jsonl(
        args.output,
        output_rows,
    )

    nonempty_samples = sum(
        row["num_candidates"] > 0
        for row in output_rows
    )

    stats = {
        "schema_version":
            "global-formula-structure-pool-v2",

        "branch_candidate_rows":
            branch_rows,

        "samples":
            len(output_rows),

        "nonempty_samples":
            nonempty_samples,

        "empty_samples":
            len(output_rows)
            - nonempty_samples,

        "raw_candidates":
            raw_candidates,

        "unique_candidates":
            unique_total,

        "duplicate_routes_merged":
            duplicate_routes,

        "invalid_smiles":
            invalid_smiles,

        "average_unique_candidates_per_sample":
            (
                unique_total
                / len(output_rows)
                if output_rows
                else 0.0
            ),

        "branch_route_order_preserved":
            True,

        "cross_formula_ranking_applied":
            False,

        "formula_confidence_used_for_ranking":
            False,
    }

    stats_path = Path(
        args.stats
    )

    stats_path.parent.mkdir(
        parents=True,
        exist_ok=True
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
        json.dumps(
            stats,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
