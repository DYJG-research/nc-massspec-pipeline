#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import importlib.util
import itertools
import json
import math
import time
from pathlib import Path
from typing import Any

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.error")
RDLogger.DisableLog("rdApp.warning")

from multiformula_common import find_forbidden_paths


def load_legacy_module():
    path = Path(__file__).resolve().with_name("80_formula_lattice_predict.py")
    if not path.is_file():
        raise RuntimeError(f"Missing frozen legacy Formula-Lattice implementation: {path}")

    spec = importlib.util.spec_from_file_location("formula_lattice_legacy_v80", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import legacy module: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


legacy = load_legacy_module()


FORMULA_DELTA_DEHYDRATE = legacy.FORMULA_DELTA_DEHYDRATE
FORMULA_DELTA_DEACETYL = legacy.FORMULA_DELTA_DEACETYL
FORMULA_DELTA_TRI_HYDROXY = legacy.FORMULA_DELTA_TRI_HYDROXY
FORMULA_DELTA_DOUBLE_REDUCTION = legacy.FORMULA_DELTA_DOUBLE_REDUCTION


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return legacy.read_jsonl(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    legacy.write_jsonl(path, rows)


def check_no_gt(rows: list[dict[str, Any]], label: str) -> None:
    for index, row in enumerate(rows, 1):
        hits = find_forbidden_paths(row)
        if hits:
            raise RuntimeError(
                f"{label} row {index} contains forbidden GT fields: {hits}"
            )


def load_current_branch_candidates(
    candidate_rows: list[dict[str, Any]],
    branch_by_id: dict[str, dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    row_by_branch = {}
    for row in candidate_rows:
        branch_id = str(row["branch_id"])
        if branch_id in row_by_branch:
            raise RuntimeError(f"Duplicate candidate row for branch_id={branch_id}")
        row_by_branch[branch_id] = row

    if set(row_by_branch) != set(branch_by_id):
        missing = sorted(set(branch_by_id) - set(row_by_branch))[:20]
        extra = sorted(set(row_by_branch) - set(branch_by_id))[:20]
        raise RuntimeError(
            "Branch mismatch between manifest and candidate file; "
            f"missing={missing} extra={extra}"
        )

    candidates_by_branch: dict[str, list[dict[str, Any]]] = {}

    for branch_id, branch in branch_by_id.items():
        row = row_by_branch[branch_id]
        target_formula = str(branch["formula"])
        output = []
        seen = set()

        for fallback_rank, candidate in enumerate(row.get("candidates") or [], 1):
            if not isinstance(candidate, dict):
                continue

            smiles = legacy.canonical_smiles(str(candidate.get("smiles") or ""))
            if smiles is None or smiles in seen:
                continue

            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                continue

            exact = legacy.canonical_exact_molecule(molecule, target_formula)
            if exact is None:
                continue

            seen.add(exact)
            output.append(
                {
                    "smiles": exact,
                    "rank": int(candidate.get("rank") or fallback_rank),
                    "generator_raw": candidate.get("generator_raw"),
                    "generator_score_name": candidate.get("generator_score_name"),
                    "sirius_scores": candidate.get("sirius_scores", {}),
                    "source": "current_branch_msnovelist",
                    "original_source": candidate.get("source"),
                }
            )

        output.sort(key=lambda x: (int(x["rank"]), x["smiles"]))
        candidates_by_branch[branch_id] = output

    return candidates_by_branch


def build_sample_rows(
    branch_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    sample_by_id: dict[str, dict[str, Any]] = {}

    for branch in branch_rows:
        sample_id = str(branch["sample_id"])
        precursor_mz = float(branch["precursor_mz"])
        peaks = branch.get("peaks") or []

        current = {
            "identifier": sample_id,
            "precursor_mz": precursor_mz,
            "peaks": peaks,
        }

        if sample_id not in sample_by_id:
            sample_by_id[sample_id] = current
            continue

        previous = sample_by_id[sample_id]
        if abs(float(previous["precursor_mz"]) - precursor_mz) > 1e-8:
            raise RuntimeError(
                f"Inconsistent precursor_mz across branches for sample {sample_id}"
            )
        if previous["peaks"] != peaks:
            raise RuntimeError(
                f"Inconsistent MS/MS peaks across branches for sample {sample_id}"
            )

    return list(sample_by_id.values()), sample_by_id


class BranchSpectralSimilarityCache:
    """Reuse one spectrum per sample while applying branch-specific adduct weighting."""

    def __init__(
        self,
        sample_rows: list[dict[str, Any]],
        branch_by_id: dict[str, dict[str, Any]],
    ) -> None:
        self.sample_by_id = {
            str(row["identifier"]): row
            for row in sample_rows
        }
        self.branch_by_id = branch_by_id

        self.peak_spectra = {
            sample_id: legacy.normalized_spectrum(row)
            for sample_id, row in self.sample_by_id.items()
        }
        self.loss_spectra = {
            sample_id: legacy.neutral_loss_spectrum(
                row,
                self.peak_spectra[sample_id],
            )
            for sample_id, row in self.sample_by_id.items()
        }
        self.peak_indices = {
            sample_id: legacy.build_mass_index(spectrum)
            for sample_id, spectrum in self.peak_spectra.items()
        }
        self.loss_indices = {
            sample_id: legacy.build_mass_index(spectrum)
            for sample_id, spectrum in self.loss_spectra.items()
        }

        self.component_cache: dict[tuple[str, str], tuple[float, float]] = {}
        self.score_cache: dict[tuple[str, str], float] = {}
        self.computed_sample_pairs = 0
        self.computed_branch_pairs = 0

    @staticmethod
    def _pair_key(left: str, right: str) -> tuple[str, str]:
        return (left, right) if left <= right else (right, left)

    def _components(self, left_sample: str, right_sample: str) -> tuple[float, float]:
        if left_sample == right_sample:
            return 1.0, 1.0

        key = self._pair_key(left_sample, right_sample)
        cached = self.component_cache.get(key)
        if cached is not None:
            return cached

        direct = legacy.greedy_cosine_match_indexed(
            self.peak_spectra[left_sample],
            self.peak_indices[right_sample],
        )
        neutral_loss = legacy.greedy_cosine_match_indexed(
            self.loss_spectra[left_sample],
            self.loss_indices[right_sample],
        )
        self.component_cache[key] = (direct, neutral_loss)
        self.computed_sample_pairs += 1
        return direct, neutral_loss

    def __getitem__(self, pair: tuple[str, str]) -> float:
        left_branch, right_branch = pair

        if left_branch == right_branch:
            return 1.0

        cache_key = (left_branch, right_branch)
        cached = self.score_cache.get(cache_key)
        if cached is not None:
            return cached

        left = self.branch_by_id[left_branch]
        right = self.branch_by_id[right_branch]
        left_sample = str(left["sample_id"])
        right_sample = str(right["sample_id"])

        direct, neutral_loss = self._components(left_sample, right_sample)
        same_adduct = str(left.get("adduct")) == str(right.get("adduct"))

        score = (
            0.70 * direct + 0.30 * neutral_loss
            if same_adduct
            else 0.25 * direct + 0.75 * neutral_loss
        )

        self.score_cache[(left_branch, right_branch)] = score
        self.score_cache[(right_branch, left_branch)] = score
        self.computed_branch_pairs += 1
        return score


def donor_metadata(
    donor_branch_id: str,
    branch_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    donor = branch_by_id[donor_branch_id]
    return {
        "donor_branch_id": donor_branch_id,
        "donor_sample_id": str(donor["sample_id"]),
        "donor_formula_rank": int(donor["formula_rank"]),
        "donor_formula_score": donor.get("formula_score"),
        "donor_formula_confidence": donor.get("formula_confidence"),
        "donor_adduct": donor.get("adduct"),
    }


def build_same_formula_transfer(
    target_branch_id: str,
    target_formula: str,
    formula_to_branches: dict[str, list[str]],
    branch_by_id: dict[str, dict[str, Any]],
    candidates_by_branch: dict[str, list[dict[str, Any]]],
    spectral_similarity: BranchSpectralSimilarityCache,
    top_k: int,
) -> list[dict[str, Any]]:
    target_sample = str(branch_by_id[target_branch_id]["sample_id"])

    donors = [
        donor_branch_id
        for donor_branch_id in formula_to_branches[target_formula]
        if candidates_by_branch[donor_branch_id]
        and (
            donor_branch_id == target_branch_id
            or str(branch_by_id[donor_branch_id]["sample_id"]) != target_sample
        )
    ]

    # V1 deliberately reproduces the legacy spectral ordering. Formula
    # confidence is propagated as provenance but is not yet used for ranking.
    donors.sort(
        key=lambda donor_branch_id: (
            donor_branch_id == target_branch_id,
            spectral_similarity[(target_branch_id, donor_branch_id)],
            donor_branch_id,
        ),
        reverse=True,
    )

    output: list[dict[str, Any]] = []
    seen: set[str] = set()

    if candidates_by_branch[target_branch_id]:
        candidate = candidates_by_branch[target_branch_id][0]
        output.append(
            {
                **candidate,
                "prediction_source": "own_top1_lock",
                "donor_identifier": target_branch_id,
                "spectral_similarity": 1.0,
                **donor_metadata(target_branch_id, branch_by_id),
            }
        )
        seen.add(candidate["smiles"])

    for rank_index in range(100):
        for donor_branch_id in donors:
            if donor_branch_id == target_branch_id and rank_index == 0:
                continue

            donor_candidates = candidates_by_branch[donor_branch_id]
            if rank_index >= len(donor_candidates):
                continue

            candidate = donor_candidates[rank_index]
            if candidate["smiles"] in seen:
                continue

            output.append(
                {
                    **candidate,
                    "prediction_source": (
                        "own_msnovelist"
                        if donor_branch_id == target_branch_id
                        else "same_formula_batch_transfer"
                    ),
                    "donor_identifier": donor_branch_id,
                    "donor_rank": int(candidate["rank"]),
                    "spectral_similarity": spectral_similarity[
                        (target_branch_id, donor_branch_id)
                    ],
                    **donor_metadata(donor_branch_id, branch_by_id),
                }
            )
            seen.add(candidate["smiles"])

            if len(output) >= top_k:
                return output

    return output


def generate_formula_lattice_candidates(
    target_branch_id: str,
    branch_by_id: dict[str, dict[str, Any]],
    formula_to_branches: dict[str, list[str]],
    formula_rules_by_target: dict[str, list[tuple[str, str]]],
    candidates_by_branch: dict[str, list[dict[str, Any]]],
    spectral_similarity: BranchSpectralSimilarityCache,
    landmarks: list[int],
    molecule_cache: dict[str, Chem.Mol | None],
    transformation_cache: dict[tuple[str, str, str], Any],
    allow_intrasample_neighbor: bool,
) -> list[dict[str, Any]]:
    target = branch_by_id[target_branch_id]
    target_formula = str(target["formula"])
    target_sample = str(target["sample_id"])
    originally_empty = not candidates_by_branch[target_branch_id]

    output: list[dict[str, Any]] = []
    seen: set[str] = set()

    formula_rules = [
        (operation, donor_formula)
        for operation, donor_formula in formula_rules_by_target.get(target_formula, [])
        if operation in {"dehydration", "deacetylation"}
        or originally_empty
    ]

    for operation, donor_formula in formula_rules:
        donor_branches = [
            donor_branch_id
            for donor_branch_id in formula_to_branches[donor_formula]
            if candidates_by_branch[donor_branch_id]
            and (
                allow_intrasample_neighbor
                or str(branch_by_id[donor_branch_id]["sample_id"]) != target_sample
            )
        ]

        donor_branches.sort(
            key=lambda donor_branch_id: (
                spectral_similarity[(target_branch_id, donor_branch_id)],
                donor_branch_id,
            ),
            reverse=True,
        )

        if operation in {"dehydration", "deacetylation"}:
            proposals = []

            for donor_branch_id in donor_branches:
                donor_similarity = spectral_similarity[
                    (target_branch_id, donor_branch_id)
                ]

                for candidate in candidates_by_branch[donor_branch_id][:100]:
                    smiles_key = candidate["smiles"]
                    cache_key = (operation, smiles_key, target_formula)

                    if cache_key in transformation_cache:
                        products = transformation_cache[cache_key]
                    else:
                        molecule = molecule_cache.get(smiles_key)
                        if smiles_key not in molecule_cache:
                            molecule = Chem.MolFromSmiles(smiles_key)
                            molecule_cache[smiles_key] = molecule

                        if molecule is None:
                            products = []
                        else:
                            products = (
                                legacy.dehydrate_terminal_alcohol(
                                    molecule,
                                    target_formula,
                                )
                                if operation == "dehydration"
                                else legacy.remove_o_acetyl_group(
                                    molecule,
                                    target_formula,
                                )
                            )
                        transformation_cache[cache_key] = products

                    for product_smiles, site_score in products:
                        if product_smiles in seen:
                            continue

                        score = (
                            3.0 * math.exp(-(int(candidate["rank"]) - 1) / 15.0)
                            + 0.2 * site_score
                            + donor_similarity
                        )

                        proposals.append(
                            {
                                "smiles": product_smiles,
                                "prediction_source": f"formula_delta_{operation}",
                                "donor_identifier": donor_branch_id,
                                "donor_formula": donor_formula,
                                "donor_rank": int(candidate["rank"]),
                                "spectral_similarity": donor_similarity,
                                "site_score": site_score,
                                "transformation_score": score,
                                **donor_metadata(donor_branch_id, branch_by_id),
                            }
                        )

            proposals.sort(
                key=lambda row: (
                    row["transformation_score"],
                    -row["donor_rank"],
                    row["smiles"],
                ),
                reverse=True,
            )

            selected_count = 0
            for proposal in proposals:
                if proposal["smiles"] in seen:
                    continue
                seen.add(proposal["smiles"])
                output.append(proposal)
                selected_count += 1
                if selected_count >= 3:
                    break

        else:
            for landmark_rank in landmarks:
                donor_choices = []

                for donor_branch_id in donor_branches:
                    donor_candidates = candidates_by_branch[donor_branch_id]
                    if landmark_rank > len(donor_candidates):
                        continue

                    candidate = donor_candidates[landmark_rank - 1]
                    smiles_key = candidate["smiles"]
                    cache_key = (operation, smiles_key, target_formula)

                    if cache_key in transformation_cache:
                        result = transformation_cache[cache_key]
                    else:
                        molecule = molecule_cache.get(smiles_key)
                        if smiles_key not in molecule_cache:
                            molecule = Chem.MolFromSmiles(smiles_key)
                            molecule_cache[smiles_key] = molecule

                        if molecule is None:
                            result = None
                        else:
                            result = (
                                legacy.best_tri_hydroxylation(
                                    molecule,
                                    target_formula,
                                )
                                if operation == "tri_hydroxylation"
                                else legacy.best_double_reduction(
                                    molecule,
                                    target_formula,
                                )
                            )
                        transformation_cache[cache_key] = result

                    if result is None:
                        continue

                    product_smiles, site_score = result
                    if product_smiles in seen:
                        continue

                    donor_similarity = spectral_similarity[
                        (target_branch_id, donor_branch_id)
                    ]
                    score = donor_similarity + 0.02 * site_score

                    donor_choices.append(
                        {
                            "smiles": product_smiles,
                            "prediction_source": f"formula_delta_{operation}",
                            "donor_identifier": donor_branch_id,
                            "donor_formula": donor_formula,
                            "donor_rank": landmark_rank,
                            "spectral_similarity": donor_similarity,
                            "site_score": site_score,
                            "transformation_score": score,
                            **donor_metadata(donor_branch_id, branch_by_id),
                        }
                    )

                if donor_choices:
                    proposal = max(
                        donor_choices,
                        key=lambda row: (
                            row["transformation_score"],
                            row["donor_identifier"],
                        ),
                    )
                    if proposal["smiles"] not in seen:
                        seen.add(proposal["smiles"])
                        output.append(proposal)

    return output


def normalize_output_candidate(
    candidate: dict[str, Any],
    target_branch: dict[str, Any],
) -> dict[str, Any]:
    item = dict(candidate)
    route = str(item.get("prediction_source") or item.get("source") or "unknown")
    item["prediction_source"] = route
    item["source"] = route
    item["target_branch_id"] = str(target_branch["branch_id"])
    item["target_sample_id"] = str(target_branch["sample_id"])
    item["target_formula"] = str(target_branch["formula"])
    item["target_formula_rank"] = int(target_branch["formula_rank"])
    item["target_formula_score"] = target_branch.get("formula_score")
    item["target_formula_confidence"] = target_branch.get("formula_confidence")
    item["target_adduct"] = target_branch.get("adduct")
    return item


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Multi-hypothesis extension of the frozen V80 Formula-Lattice. "
            "Each (sample, predicted formula) branch is treated as an analytical state."
        )
    )
    parser.add_argument("--branches", required=True)
    parser.add_argument("--branch-candidates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--landmarks",
        default="1,2,3,5,7,10,14,20,30,50",
    )
    parser.add_argument("--expected-branches", type=int, required=True)
    parser.add_argument("--expected-samples", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument(
        "--lock-mode",
        choices=("top1", "top10"),
        default="top1",
    )
    parser.add_argument(
        "--allow-intrasample-neighbor",
        action="store_true",
        help=(
            "Allow formula-neighbor graph edits to use another formula branch "
            "from the same spectrum. Disabled by default to preserve the legacy "
            "cross-spectrum transductive semantics."
        ),
    )
    args = parser.parse_args()

    branch_rows = read_jsonl(Path(args.branches))
    candidate_rows = read_jsonl(Path(args.branch_candidates))

    check_no_gt(branch_rows, "branch manifest")
    check_no_gt(candidate_rows, "branch candidate file")

    if len(branch_rows) != args.expected_branches:
        raise RuntimeError(
            f"Expected {args.expected_branches} branches, found {len(branch_rows)}"
        )

    required_branch_fields = {
        "sample_id",
        "branch_id",
        "formula_rank",
        "formula",
        "adduct",
        "precursor_mz",
        "peaks",
    }

    branch_by_id: dict[str, dict[str, Any]] = {}
    order: list[str] = []

    for row in branch_rows:
        missing = required_branch_fields - set(row)
        if missing:
            raise RuntimeError(
                f"{row.get('branch_id', '<unknown>')}: missing fields {sorted(missing)}"
            )

        branch_id = str(row["branch_id"])
        if branch_id in branch_by_id:
            raise RuntimeError(f"Duplicate branch_id: {branch_id}")

        branch_by_id[branch_id] = row
        order.append(branch_id)

    sample_rows, sample_by_id = build_sample_rows(branch_rows)
    sample_count = len(sample_rows)

    if args.expected_samples is not None and sample_count != args.expected_samples:
        raise RuntimeError(
            f"Expected {args.expected_samples} samples, found {sample_count}"
        )

    landmarks = [
        int(value)
        for value in str(args.landmarks).split(",")
        if value.strip()
    ]
    if not landmarks:
        raise RuntimeError("At least one rank landmark is required")

    candidates_by_branch = load_current_branch_candidates(
        candidate_rows,
        branch_by_id,
    )

    formula_to_branches: dict[str, list[str]] = collections.defaultdict(list)
    for branch_id in order:
        formula_to_branches[str(branch_by_id[branch_id]["formula"])].append(branch_id)

    signature_to_formulas: dict[
        tuple[tuple[str, int], ...],
        list[str],
    ] = collections.defaultdict(list)

    for formula in formula_to_branches:
        signature_to_formulas[legacy.formula_signature(formula)].append(formula)

    operation_deltas = (
        ("dehydration", FORMULA_DELTA_DEHYDRATE),
        ("deacetylation", FORMULA_DELTA_DEACETYL),
        ("tri_hydroxylation", FORMULA_DELTA_TRI_HYDROXY),
        ("double_reduction", FORMULA_DELTA_DOUBLE_REDUCTION),
    )

    formula_rules_by_target: dict[str, list[tuple[str, str]]] = {}

    for target_formula in formula_to_branches:
        target_signature = legacy.formula_signature(target_formula)
        rules: list[tuple[str, str]] = []

        for operation, delta in operation_deltas:
            donor_signature = legacy.donor_signature_for_delta(
                target_signature,
                delta,
            )
            if donor_signature is None:
                continue

            for donor_formula in signature_to_formulas.get(donor_signature, []):
                if any(
                    candidates_by_branch[donor_branch_id]
                    for donor_branch_id in formula_to_branches[donor_formula]
                ):
                    rules.append((operation, donor_formula))

        formula_rules_by_target[target_formula] = rules

    print(
        "[INFO] Multi-hypothesis formula-lattice index built: "
        f"samples={sample_count} branches={len(order)} "
        f"unique_formulas={len(formula_to_branches)} "
        f"active_links={sum(len(v) for v in formula_rules_by_target.values())}",
        flush=True,
    )
    print(
        "[INFO] Formula confidence is propagated as provenance only; "
        "it is NOT used for ranking in V1.",
        flush=True,
    )

    spectral_similarity = BranchSpectralSimilarityCache(
        sample_rows,
        branch_by_id,
    )

    output_rows = []
    source_counts: collections.Counter[str] = collections.Counter()
    originally_nonempty = 0
    originally_empty = 0
    own_top1_changed = 0
    own_top10_missing_rows = 0
    molecule_cache: dict[str, Chem.Mol | None] = {}
    transformation_cache: dict[tuple[str, str, str], Any] = {}
    start_time = time.perf_counter()

    for index, branch_id in enumerate(order, 1):
        branch = branch_by_id[branch_id]
        formula = str(branch["formula"])
        nonempty = bool(candidates_by_branch[branch_id])
        originally_nonempty += int(nonempty)
        originally_empty += int(not nonempty)

        transfer_candidates = build_same_formula_transfer(
            target_branch_id=branch_id,
            target_formula=formula,
            formula_to_branches=formula_to_branches,
            branch_by_id=branch_by_id,
            candidates_by_branch=candidates_by_branch,
            spectral_similarity=spectral_similarity,
            top_k=args.top_k,
        )

        transformed_candidates = generate_formula_lattice_candidates(
            target_branch_id=branch_id,
            branch_by_id=branch_by_id,
            formula_to_branches=formula_to_branches,
            formula_rules_by_target=formula_rules_by_target,
            candidates_by_branch=candidates_by_branch,
            spectral_similarity=spectral_similarity,
            landmarks=landmarks,
            molecule_cache=molecule_cache,
            transformation_cache=transformation_cache,
            allow_intrasample_neighbor=args.allow_intrasample_neighbor,
        )

        final_candidates = legacy.merge_candidate_routes(
            original_nonempty=nonempty,
            original_candidates=candidates_by_branch[branch_id],
            transfer_candidates=transfer_candidates,
            transformed_candidates=transformed_candidates,
            top_k=args.top_k,
            lock_mode=args.lock_mode,
        )

        if nonempty:
            original_first = candidates_by_branch[branch_id][0]["smiles"]
            final_first = final_candidates[0]["smiles"] if final_candidates else None
            own_top1_changed += int(original_first != final_first)

            original_topk = {
                candidate["smiles"]
                for candidate in candidates_by_branch[branch_id][:args.top_k]
            }
            final_topk = {
                candidate["smiles"]
                for candidate in final_candidates[:args.top_k]
            }
            own_top10_missing_rows += int(not original_topk.issubset(final_topk))

        final_candidates = [
            normalize_output_candidate(candidate, branch)
            for candidate in final_candidates
        ]

        for candidate in final_candidates:
            source_counts[candidate["prediction_source"]] += 1

        output_rows.append(
            {
                "schema_version": "multihyp-formula-lattice-branch-v1",
                "sample_id": str(branch["sample_id"]),
                "branch_id": branch_id,
                "formula_rank": int(branch["formula_rank"]),
                "formula": formula,
                "adduct": branch.get("adduct"),
                "formula_score": branch.get("formula_score"),
                "formula_confidence": branch.get("formula_confidence"),
                "formula_ppm_error": branch.get("formula_ppm_error"),
                "formula_dbe": branch.get("formula_dbe"),
                "status": "ok" if final_candidates else "no_candidates",
                "originally_nonempty": nonempty,
                "num_candidates": len(final_candidates),
                "predictions": [candidate["smiles"] for candidate in final_candidates],
                "candidates": final_candidates,
                "runtime_database_access": False,
                "external_structure_retrieval": False,
                "train_fold_structure_retrieval": False,
                "llm_used": False,
                "formula_confidence_used_for_ranking": False,
                "method": (
                    "multi-hypothesis current-batch same-formula spectral transfer + "
                    "formula-lattice chemical editing + "
                    f"{args.lock_mode} within-branch accuracy lock"
                ),
            }
        )

        if (
            index == 1
            or index % max(args.progress_every, 1) == 0
            or index == len(order)
        ):
            elapsed = time.perf_counter() - start_time
            rate = index / elapsed if elapsed > 0 else 0.0
            eta = (len(order) - index) / rate if rate > 0 else float("inf")
            print(
                f"[PREDICT] {index}/{len(order)} {branch_id} "
                f"sample={branch['sample_id']} formula={formula} "
                f"formula_rank={branch['formula_rank']} "
                f"base={len(candidates_by_branch[branch_id])} "
                f"same_formula={len(transfer_candidates)} "
                f"transforms={len(transformed_candidates)} "
                f"final={len(final_candidates)} "
                f"rate={rate:.2f}/s eta={eta/60:.1f}m",
                flush=True,
            )

    output_path = Path(args.output)
    write_jsonl(output_path, output_rows)

    audit = {
        "num_samples": sample_count,
        "num_branches": len(output_rows),
        "originally_nonempty_branches": originally_nonempty,
        "originally_empty_branches": originally_empty,
        "branches_with_final_candidates": sum(
            bool(row["predictions"])
            for row in output_rows
        ),
        "own_top1_changed_branches": own_top1_changed,
        "own_top10_missing_branches": own_top10_missing_rows,
        "lock_mode": args.lock_mode,
        "candidate_source_counts": dict(source_counts),
        "database_access": False,
        "external_candidate_files": [],
        "truth_fields_read_by_predictor": [],
        "formula_confidence_propagated": True,
        "formula_confidence_used_for_ranking": False,
        "intrasample_formula_neighbor_enabled": bool(args.allow_intrasample_neighbor),
        "landmark_ranks": landmarks,
        "expected_branches": args.expected_branches,
        "spectral_sample_pairs_computed": spectral_similarity.computed_sample_pairs,
        "spectral_branch_pairs_computed": spectral_similarity.computed_branch_pairs,
        "unique_formula_count": len(formula_to_branches),
        "formula_lattice_active_links": sum(
            len(value) for value in formula_rules_by_target.values()
        ),
        "molecule_cache_size": len(molecule_cache),
        "transformation_cache_size": len(transformation_cache),
        "elapsed_seconds": time.perf_counter() - start_time,
        "candidate_universe": (
            "Only MSNovelist candidates generated for predicted-formula branches "
            f"in the current {sample_count}-sample batch, plus deterministic RDKit "
            "formula-delta edits."
        ),
    }

    audit_path = Path(args.audit)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if own_top1_changed:
        raise RuntimeError(
            f"Within-branch Top-1 lock failed for {own_top1_changed} nonempty branches"
        )

    if args.lock_mode == "top10" and own_top10_missing_rows:
        raise RuntimeError(
            "Within-branch Top-10 lock failed for "
            f"{own_top10_missing_rows} nonempty branches"
        )

    print("[OK] Multi-hypothesis Formula-Lattice output:", output_path)
    print("[OK] Audit:", audit_path)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
