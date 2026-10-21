#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import collections
import itertools
import json
import math
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

RDLogger.DisableLog("rdApp.error")
RDLogger.DisableLog("rdApp.warning")

FORMULA_DELTA_DEHYDRATE = (("H", -2), ("O", -1))
FORMULA_DELTA_DEACETYL = (("C", -2), ("H", -2), ("O", -1))
FORMULA_DELTA_TRI_HYDROXY = (("O", 3),)
FORMULA_DELTA_DOUBLE_REDUCTION = (("H", 4),)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise RuntimeError(f"{path}:{line_number}: JSON row is not an object")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


CHARGE_KEY = "__formal_charge__"


def split_formula_charge(formula: str) -> tuple[str, int]:
    """Split an RDKit-style molecular formula into neutral atom formula and charge.

    Supported RDKit suffixes include ``+``, ``-``, ``+2`` and ``-2``.
    RDKit normally emits forms such as ``C22H28N7O+`` and ``C2H10N2+2``.
    """
    compact = re.sub(r"\s+", "", str(formula))
    if not compact:
        raise ValueError("Empty molecular formula")

    # RDKit emits charge suffixes with the sign first: ``+``, ``-``,
    # ``+2`` or ``-2``. We intentionally do not interpret an ambiguous
    # trailing form such as ``2-`` as charge magnitude because the ``2`` may
    # be the atom count of the last element (for example ``CHO2-``).
    match = re.search(r"(?P<sign>[+-])(?P<magnitude>\d*)$", compact)
    if match is None:
        return compact, 0

    sign = 1 if match.group("sign") == "+" else -1
    magnitude = int(match.group("magnitude") or "1")
    core = compact[: match.start()]
    if not core:
        raise ValueError(f"Unsupported molecular formula: {formula}")
    return core, sign * magnitude


def parse_formula_components(formula: str) -> tuple[dict[str, int], int]:
    core, charge = split_formula_charge(formula)
    parts = re.findall(r"([A-Z][a-z]?)(\d*)", core)
    if "".join(element + count for element, count in parts) != core:
        raise ValueError(f"Unsupported molecular formula: {formula}")

    counts: dict[str, int] = {}
    for element, count in parts:
        counts[element] = counts.get(element, 0) + int(count or "1")
    return counts, charge


def parse_formula(formula: str) -> dict[str, int]:
    counts, _ = parse_formula_components(formula)
    return counts


@lru_cache(maxsize=None)
def formula_signature(formula: str) -> tuple[tuple[str, int], ...]:
    counts, charge = parse_formula_components(formula)
    signature = dict(counts)
    signature[CHARGE_KEY] = charge
    return tuple(sorted(signature.items()))


def donor_signature_for_delta(
    target_signature: tuple[tuple[str, int], ...],
    delta: tuple[tuple[str, int], ...],
) -> tuple[tuple[str, int], ...] | None:
    # formula_delta(target, donor) == delta, therefore donor = target - delta.
    counts = dict(target_signature)
    for element, change in delta:
        counts[element] = counts.get(element, 0) - change
        if counts[element] < 0:
            return None
        if counts[element] == 0:
            counts.pop(element)
    return tuple(sorted(counts.items()))


def formula_delta(target: str, donor: str) -> tuple[tuple[str, int], ...]:
    target_counts = parse_formula(target)
    donor_counts = parse_formula(donor)
    delta = []
    for element in sorted(set(target_counts) | set(donor_counts)):
        value = target_counts.get(element, 0) - donor_counts.get(element, 0)
        if value:
            delta.append((element, value))
    return tuple(delta)


def canonical_smiles(smiles: str) -> str | None:
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        return None
    return Chem.MolToSmiles(
        molecule,
        canonical=True,
        isomericSmiles=False,
    )


def canonical_exact_molecule(molecule: Chem.Mol, target_formula: str) -> str | None:
    try:
        Chem.SanitizeMol(molecule)
    except Exception:
        return None
    if len(Chem.GetMolFrags(molecule)) != 1:
        return None
    if any(atom.GetNumRadicalElectrons() for atom in molecule.GetAtoms()):
        return None

    try:
        target_counts, target_charge = parse_formula_components(target_formula)
        actual_formula = rdMolDescriptors.CalcMolFormula(molecule)
        actual_counts, actual_charge = parse_formula_components(actual_formula)
    except ValueError:
        return None

    if actual_counts != target_counts or actual_charge != target_charge:
        return None
    if Chem.GetFormalCharge(molecule) != target_charge:
        return None

    return Chem.MolToSmiles(
        molecule,
        canonical=True,
        isomericSmiles=False,
    )


def normalized_spectrum(row: dict[str, Any]) -> list[tuple[float, float]]:
    precursor = float(row["precursor_mz"])
    peaks = []
    for pair in row.get("peaks") or []:
        if not isinstance(pair, (list, tuple)) or len(pair) < 2:
            continue
        try:
            mz = float(pair[0])
            intensity = float(pair[1])
        except (TypeError, ValueError):
            continue
        if abs(mz - precursor) < 0.5:
            continue
        value = math.sqrt(max(intensity, 0.0))
        if value > 0:
            peaks.append((mz, value))
    norm = math.sqrt(sum(value * value for _, value in peaks)) or 1.0
    return [(mz, value / norm) for mz, value in peaks]


def neutral_loss_spectrum(
    row: dict[str, Any],
    normalized_peaks: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    precursor = float(row["precursor_mz"])
    return sorted(
        (precursor - mz, intensity)
        for mz, intensity in normalized_peaks
        if precursor - mz > 1.0
    )


def build_mass_index(
    spectrum: list[tuple[float, float]],
) -> tuple[list[float], list[tuple[int, float, float]]]:
    indexed = sorted(
        (mass, original_index, intensity)
        for original_index, (mass, intensity) in enumerate(spectrum)
    )
    return (
        [mass for mass, _, _ in indexed],
        [(original_index, mass, intensity) for mass, original_index, intensity in indexed],
    )


def greedy_cosine_match_indexed(
    spectrum_a: list[tuple[float, float]],
    spectrum_b_index: tuple[list[float], list[tuple[int, float, float]]],
    ppm: float = 20.0,
    absolute_tolerance: float = 0.025,
) -> float:
    # Exact replacement for the original nested-loop matcher. Bisect only
    # prunes masses that cannot satisfy the same tolerance test.
    masses_b, indexed_b = spectrum_b_index
    proposals = []
    ppm_fraction = ppm / 1e6

    for index_a, (mass_a, intensity_a) in enumerate(spectrum_a):
        broad_tolerance = max(
            absolute_tolerance,
            mass_a * ppm_fraction / max(1.0 - ppm_fraction, 1e-12),
        )
        left = bisect.bisect_left(masses_b, mass_a - broad_tolerance)
        right = bisect.bisect_right(masses_b, mass_a + broad_tolerance)

        for position in range(left, right):
            index_b, mass_b, intensity_b = indexed_b[position]
            tolerance = max(
                absolute_tolerance,
                mass_a * ppm_fraction,
                mass_b * ppm_fraction,
            )
            if abs(mass_a - mass_b) <= tolerance:
                proposals.append(
                    (intensity_a * intensity_b, index_a, index_b)
                )

    proposals.sort(reverse=True)
    used_a: set[int] = set()
    used_b: set[int] = set()
    score = 0.0
    for product, index_a, index_b in proposals:
        if index_a in used_a or index_b in used_b:
            continue
        used_a.add(index_a)
        used_b.add(index_b)
        score += product
    return score


class SpectralSimilarityCache:
    """Compute only the spectrum pairs actually requested by the lattice search."""

    def __init__(self, selected_rows: list[dict[str, Any]]) -> None:
        self.by_id = {
            str(row["identifier"]): row
            for row in selected_rows
        }
        self.peak_spectra = {
            identifier: normalized_spectrum(row)
            for identifier, row in self.by_id.items()
        }
        self.loss_spectra = {
            identifier: neutral_loss_spectrum(
                row,
                self.peak_spectra[identifier],
            )
            for identifier, row in self.by_id.items()
        }
        self.peak_indices = {
            identifier: build_mass_index(spectrum)
            for identifier, spectrum in self.peak_spectra.items()
        }
        self.loss_indices = {
            identifier: build_mass_index(spectrum)
            for identifier, spectrum in self.loss_spectra.items()
        }
        self.cache: dict[tuple[str, str], float] = {}
        self.computed_pairs = 0

    def __getitem__(self, pair: tuple[str, str]) -> float:
        left, right = pair
        cached = self.cache.get((left, right))
        if cached is not None:
            return cached

        if left == right:
            self.cache[(left, right)] = 1.0
            return 1.0

        direct = greedy_cosine_match_indexed(
            self.peak_spectra[left],
            self.peak_indices[right],
        )
        neutral_loss = greedy_cosine_match_indexed(
            self.loss_spectra[left],
            self.loss_indices[right],
        )
        same_adduct = str(self.by_id[left].get("adduct")) == str(
            self.by_id[right].get("adduct")
        )
        score = (
            0.70 * direct + 0.30 * neutral_loss
            if same_adduct
            else 0.25 * direct + 0.75 * neutral_loss
        )
        self.cache[(left, right)] = score
        self.cache[(right, left)] = score
        self.computed_pairs += 1
        return score


def load_current_batch_candidates(
    denovo_rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    candidates_by_id: dict[str, list[dict[str, Any]]] = {}
    for row in denovo_rows:
        identifier = str(row["identifier"])
        output = []
        seen = set()
        for fallback_rank, candidate in enumerate(row.get("candidates") or [], 1):
            if not isinstance(candidate, dict):
                continue
            smiles = canonical_smiles(str(candidate.get("smiles") or ""))
            if smiles is None or smiles in seen:
                continue
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                continue
            if rdMolDescriptors.CalcMolFormula(molecule) != str(row["formula"]):
                continue
            seen.add(smiles)
            output.append(
                {
                    "smiles": smiles,
                    "rank": int(candidate.get("rank") or fallback_rank),
                    "generator_raw": candidate.get("generator_raw"),
                    "generator_score_name": candidate.get(
                        "generator_score_name"
                    ),
                    "source": "current_batch_msnovelist",
                    "original_source": candidate.get("source"),
                }
            )
        candidates_by_id[identifier] = output
    return candidates_by_id


def build_same_formula_transfer(
    target_id: str,
    target_formula: str,
    formula_to_ids: dict[str, list[str]],
    candidates_by_id: dict[str, list[dict[str, Any]]],
    spectral_similarity: SpectralSimilarityCache,
    top_k: int,
) -> list[dict[str, Any]]:
    donors = [
        donor_id
        for donor_id in formula_to_ids[target_formula]
        if candidates_by_id[donor_id]
    ]
    donors.sort(
        key=lambda donor_id: (
            donor_id == target_id,
            spectral_similarity[(target_id, donor_id)],
            donor_id,
        ),
        reverse=True,
    )

    output: list[dict[str, Any]] = []
    seen: set[str] = set()

    # Top-1 lock: the original top candidate is immutable for every nonempty row.
    if candidates_by_id[target_id]:
        candidate = candidates_by_id[target_id][0]
        output.append(
            {
                **candidate,
                "prediction_source": "own_top1_lock",
                "donor_identifier": target_id,
                "spectral_similarity": 1.0,
            }
        )
        seen.add(candidate["smiles"])

    for rank_index in range(100):
        for donor_id in donors:
            if donor_id == target_id and rank_index == 0:
                continue
            donor_candidates = candidates_by_id[donor_id]
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
                        if donor_id == target_id
                        else "same_formula_batch_transfer"
                    ),
                    "donor_identifier": donor_id,
                    "donor_rank": int(candidate["rank"]),
                    "spectral_similarity": spectral_similarity[
                        (target_id, donor_id)
                    ],
                }
            )
            seen.add(candidate["smiles"])
            if len(output) >= top_k:
                return output
    return output


def dehydrate_terminal_alcohol(
    molecule: Chem.Mol,
    target_formula: str,
) -> list[tuple[str, float]]:
    products = []
    for oxygen in molecule.GetAtoms():
        if (
            oxygen.GetSymbol() != "O"
            or oxygen.GetDegree() != 1
            or oxygen.GetTotalNumHs() < 1
        ):
            continue
        carbon = oxygen.GetNeighbors()[0]
        if carbon.GetSymbol() != "C":
            continue
        for bond in carbon.GetBonds():
            neighbor = bond.GetOtherAtom(carbon)
            if neighbor.GetIdx() == oxygen.GetIdx():
                continue
            if bond.GetBondType() != Chem.BondType.SINGLE:
                continue
            if neighbor.GetSymbol() != "C" or neighbor.GetTotalNumHs() < 1:
                continue

            editable = Chem.RWMol(molecule)
            oxygen_index = oxygen.GetIdx()
            carbon_index = carbon.GetIdx()
            neighbor_index = neighbor.GetIdx()
            editable.RemoveBond(carbon_index, neighbor_index)
            editable.RemoveAtom(oxygen_index)
            carbon_index -= int(oxygen_index < carbon_index)
            neighbor_index -= int(oxygen_index < neighbor_index)
            editable.AddBond(
                carbon_index,
                neighbor_index,
                Chem.BondType.DOUBLE,
            )
            smiles = canonical_exact_molecule(
                editable.GetMol(),
                target_formula,
            )
            if smiles is None:
                continue

            site_score = (
                2.0 * int(not carbon.IsInRing())
                + 2.0 * int(not neighbor.IsInRing())
                + 2.0 * int(carbon.GetDegree() >= 3)
                + 2.0 * int(neighbor.GetDegree() == 1)
                + 1.0 * int(neighbor.GetTotalNumHs() >= 2)
            )
            products.append((smiles, site_score))
    return products


def remove_o_acetyl_group(
    molecule: Chem.Mol,
    target_formula: str,
) -> list[tuple[str, float]]:
    products = []
    for carbonyl_carbon in molecule.GetAtoms():
        if carbonyl_carbon.GetSymbol() != "C":
            continue
        double_oxygen = []
        single_neighbors = []
        for bond in carbonyl_carbon.GetBonds():
            neighbor = bond.GetOtherAtom(carbonyl_carbon)
            if (
                neighbor.GetSymbol() == "O"
                and bond.GetBondType() == Chem.BondType.DOUBLE
            ):
                double_oxygen.append(neighbor)
            elif bond.GetBondType() == Chem.BondType.SINGLE:
                single_neighbors.append(neighbor)

        if len(double_oxygen) != 1 or len(single_neighbors) != 2:
            continue
        core_hetero = [
            neighbor
            for neighbor in single_neighbors
            if neighbor.GetSymbol() in {"O", "N", "S"}
        ]
        methyl = [
            neighbor
            for neighbor in single_neighbors
            if neighbor.GetSymbol() == "C" and neighbor.GetDegree() == 1
        ]
        if not core_hetero or not methyl:
            continue

        editable = Chem.RWMol(molecule)
        remove_indices = sorted(
            [
                carbonyl_carbon.GetIdx(),
                double_oxygen[0].GetIdx(),
                methyl[0].GetIdx(),
            ],
            reverse=True,
        )
        for atom_index in remove_indices:
            editable.RemoveAtom(atom_index)
        smiles = canonical_exact_molecule(editable.GetMol(), target_formula)
        if smiles is not None:
            products.append((smiles, 5.0))
    return products


def best_tri_hydroxylation(
    molecule: Chem.Mol,
    target_formula: str,
) -> tuple[str, float] | None:
    sites = []
    for atom in molecule.GetAtoms():
        if atom.GetSymbol() != "C" or atom.GetTotalNumHs() < 1:
            continue
        features = {
            "atom_index": atom.GetIdx(),
            "ring": int(atom.IsInRing()),
            "degree": atom.GetDegree(),
            "hydrogens": atom.GetTotalNumHs(),
            "aromatic": int(atom.GetIsAromatic()),
            "sp3": int(str(atom.GetHybridization()) == "SP3"),
            "hetero_neighbors": sum(
                neighbor.GetSymbol() != "C"
                for neighbor in atom.GetNeighbors()
            ),
        }
        features["site_score"] = (
            3.0 * features["ring"]
            + 0.5 * features["degree"]
            + 0.5 * features["hydrogens"]
            + 2.0 * features["sp3"]
            - 1.0 * features["hetero_neighbors"]
            - 5.0 * features["aromatic"]
        )
        sites.append(features)

    combinations = sorted(
        itertools.combinations(sites, 3),
        key=lambda group: sum(site["site_score"] for site in group),
        reverse=True,
    )
    # Only the best chemistry-prior combinations are materialized.
    for group in combinations[:100]:
        editable = Chem.RWMol(molecule)
        for site in group:
            oxygen_index = editable.AddAtom(Chem.Atom("O"))
            editable.AddBond(
                int(site["atom_index"]),
                oxygen_index,
                Chem.BondType.SINGLE,
            )
        smiles = canonical_exact_molecule(editable.GetMol(), target_formula)
        if smiles is not None:
            return smiles, sum(site["site_score"] for site in group)
    return None


def best_double_reduction(
    molecule: Chem.Mol,
    target_formula: str,
) -> tuple[str, float] | None:
    multiple_bonds = [
        bond
        for bond in molecule.GetBonds()
        if not bond.GetIsAromatic() and bond.GetBondTypeAsDouble() >= 2
    ]
    bond_pairs = []
    for first, second in itertools.combinations(multiple_bonds, 2):
        site_score = 0.0
        for bond in (first, second):
            left = bond.GetBeginAtom()
            right = bond.GetEndAtom()
            atom_pair = "".join(sorted((left.GetSymbol(), right.GetSymbol())))
            site_score += 0.5 * int(atom_pair == "CO")
            site_score += 1.0 * int(not bond.IsInRing())
            site_score += 0.2 * int(bond.GetIsConjugated())
        bond_pairs.append((site_score, first, second))
    bond_pairs.sort(key=lambda item: item[0], reverse=True)

    for site_score, first, second in bond_pairs[:100]:
        editable = Chem.RWMol(molecule)
        for bond in (first, second):
            left_index = bond.GetBeginAtomIdx()
            right_index = bond.GetEndAtomIdx()
            original_order = int(bond.GetBondTypeAsDouble())
            editable.RemoveBond(left_index, right_index)
            editable.AddBond(
                left_index,
                right_index,
                (
                    Chem.BondType.SINGLE
                    if original_order == 2
                    else Chem.BondType.DOUBLE
                ),
            )
        smiles = canonical_exact_molecule(editable.GetMol(), target_formula)
        if smiles is not None:
            return smiles, site_score
    return None


def generate_formula_lattice_candidates(
    target_id: str,
    selected_by_id: dict[str, dict[str, Any]],
    formula_to_ids: dict[str, list[str]],
    formula_rules_by_target: dict[str, list[tuple[str, str]]],
    candidates_by_id: dict[str, list[dict[str, Any]]],
    spectral_similarity: SpectralSimilarityCache,
    landmarks: list[int],
    molecule_cache: dict[str, Chem.Mol | None],
    transformation_cache: dict[tuple[str, str, str], Any],
) -> list[dict[str, Any]]:
    target_formula = str(selected_by_id[target_id]["true_formula"])
    originally_empty = not candidates_by_id[target_id]
    output: list[dict[str, Any]] = []
    seen: set[str] = set()

    formula_rules = [
        (operation, donor_formula)
        for operation, donor_formula in formula_rules_by_target.get(
            target_formula,
            [],
        )
        if operation in {"dehydration", "deacetylation"}
        or originally_empty
    ]

    for operation, donor_formula in formula_rules:
        donor_ids = [
            donor_id
            for donor_id in formula_to_ids[donor_formula]
            if candidates_by_id[donor_id]
        ]
        donor_ids.sort(
            key=lambda donor_id: (
                spectral_similarity[(target_id, donor_id)],
                donor_id,
            ),
            reverse=True,
        )

        if operation in {"dehydration", "deacetylation"}:
            proposals = []
            for donor_id in donor_ids:
                donor_similarity = spectral_similarity[(target_id, donor_id)]
                for candidate in candidates_by_id[donor_id][:100]:
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
                                dehydrate_terminal_alcohol(molecule, target_formula)
                                if operation == "dehydration"
                                else remove_o_acetyl_group(molecule, target_formula)
                            )
                        transformation_cache[cache_key] = products

                    for product_smiles, site_score in products:
                        if product_smiles in seen:
                            continue
                        score = (
                            3.0
                            * math.exp(-(int(candidate["rank"]) - 1) / 15.0)
                            + 0.2 * site_score
                            + donor_similarity
                        )
                        proposals.append(
                            {
                                "smiles": product_smiles,
                                "prediction_source": f"formula_delta_{operation}",
                                "donor_identifier": donor_id,
                                "donor_formula": donor_formula,
                                "donor_rank": int(candidate["rank"]),
                                "spectral_similarity": donor_similarity,
                                "site_score": site_score,
                                "transformation_score": score,
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
                for donor_id in donor_ids:
                    donor_candidates = candidates_by_id[donor_id]
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
                                best_tri_hydroxylation(molecule, target_formula)
                                if operation == "tri_hydroxylation"
                                else best_double_reduction(molecule, target_formula)
                            )
                        transformation_cache[cache_key] = result

                    if result is None:
                        continue
                    product_smiles, site_score = result
                    if product_smiles in seen:
                        continue
                    donor_similarity = spectral_similarity[(target_id, donor_id)]
                    score = donor_similarity + 0.02 * site_score
                    donor_choices.append(
                        {
                            "smiles": product_smiles,
                            "prediction_source": f"formula_delta_{operation}",
                            "donor_identifier": donor_id,
                            "donor_formula": donor_formula,
                            "donor_rank": landmark_rank,
                            "spectral_similarity": donor_similarity,
                            "site_score": site_score,
                            "transformation_score": score,
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


def merge_candidate_routes(
    original_nonempty: bool,
    original_candidates: list[dict[str, Any]],
    transfer_candidates: list[dict[str, Any]],
    transformed_candidates: list[dict[str, Any]],
    top_k: int,
    lock_mode: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()

    if lock_mode == "top10" and original_nonempty:
        # Strict safety mode: preserve every original candidate that was
        # already inside the evaluated Top-K, in the same order.
        for candidate in original_candidates[:top_k]:
            item = {
                **candidate,
                "prediction_source": "own_top10_lock",
                "donor_identifier": None,
                "spectral_similarity": 1.0,
            }
            if item["smiles"] in seen:
                continue
            output.append(item)
            seen.add(item["smiles"])
        if len(output) >= top_k:
            return output

        # Only unused slots may be filled.
        additions = transformed_candidates + transfer_candidates
        for candidate in additions:
            if candidate["smiles"] in seen:
                continue
            output.append(candidate)
            seen.add(candidate["smiles"])
            if len(output) >= top_k:
                return output
        return output

    # Frozen V8 policy: immutable original Top-1, then explicit formula edits,
    # then same-formula transfer. The evaluator still rejects any run whose
    # actual Top-1 or Top-10 accuracy decreases.
    if transfer_candidates:
        first = transfer_candidates[0]
        output.append(first)
        seen.add(first["smiles"])

    high_confidence = [
        row
        for row in transformed_candidates
        if row["prediction_source"]
        in {
            "formula_delta_dehydration",
            "formula_delta_deacetylation",
        }
    ]
    multi_edit = [
        row
        for row in transformed_candidates
        if row["prediction_source"]
        in {
            "formula_delta_tri_hydroxylation",
            "formula_delta_double_reduction",
        }
    ]

    proposals = high_confidence[:1] + multi_edit
    for proposal in proposals:
        if proposal["smiles"] in seen:
            continue
        output.append(proposal)
        seen.add(proposal["smiles"])
        if len(output) >= top_k:
            return output

    transfer_tail = transfer_candidates[1:] if transfer_candidates else []
    for candidate in transfer_tail:
        if candidate["smiles"] in seen:
            continue
        output.append(candidate)
        seen.add(candidate["smiles"])
        if len(output) >= top_k:
            return output

    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Database-free transductive candidate transfer and formula-lattice "
            "chemical editing for a fixed LC-MS/MS batch."
        )
    )
    parser.add_argument("--selected", required=True)
    parser.add_argument("--denovo-candidates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--landmarks",
        default="1,2,3,5,7,10,14,20,30,50",
    )
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Print progress every N rows instead of flushing one line per sample.",
    )
    parser.add_argument(
        "--lock-mode",
        choices=("top1", "top10"),
        default="top1",
        help=(
            "top1 reproduces the frozen V8 output policy; top10 strictly "
            "preserves the complete original evaluated Top-K."
        ),
    )
    args = parser.parse_args()

    selected_rows = read_jsonl(Path(args.selected))
    denovo_rows = read_jsonl(Path(args.denovo_candidates))
    selected_by_id = {
        str(row["identifier"]): row
        for row in selected_rows
    }
    denovo_by_id = {
        str(row["identifier"]): row
        for row in denovo_rows
    }
    order = [str(row["identifier"]) for row in selected_rows]

    if len(order) != args.expected_rows:
        raise RuntimeError(
            f"Expected {args.expected_rows} selected rows, found {len(order)}"
        )
    if set(selected_by_id) != set(denovo_by_id):
        raise RuntimeError("Identifier mismatch between selected and candidate files")

    # Prediction code never reads true_smiles, canonical2d or inchikey2d.
    required_selected_fields = {
        "identifier",
        "true_formula",
        "precursor_mz",
        "adduct",
        "peaks",
    }
    for identifier, row in selected_by_id.items():
        missing = required_selected_fields - set(row)
        if missing:
            raise RuntimeError(f"{identifier}: missing fields {sorted(missing)}")

    landmarks = [
        int(value)
        for value in str(args.landmarks).split(",")
        if value.strip()
    ]
    if not landmarks:
        raise RuntimeError("At least one rank landmark is required")

    candidates_by_id = load_current_batch_candidates(denovo_rows)
    formula_to_ids: dict[str, list[str]] = collections.defaultdict(list)
    for identifier in order:
        formula_to_ids[str(selected_by_id[identifier]["true_formula"])].append(
            identifier
        )

    signature_to_formulas: dict[
        tuple[tuple[str, int], ...],
        list[str],
    ] = collections.defaultdict(list)
    for formula in formula_to_ids:
        signature_to_formulas[formula_signature(formula)].append(formula)

    operation_deltas = (
        ("dehydration", FORMULA_DELTA_DEHYDRATE),
        ("deacetylation", FORMULA_DELTA_DEACETYL),
        ("tri_hydroxylation", FORMULA_DELTA_TRI_HYDROXY),
        ("double_reduction", FORMULA_DELTA_DOUBLE_REDUCTION),
    )
    formula_rules_by_target: dict[str, list[tuple[str, str]]] = {}
    for target_formula in formula_to_ids:
        target_signature = formula_signature(target_formula)
        rules: list[tuple[str, str]] = []
        for operation, delta in operation_deltas:
            donor_signature = donor_signature_for_delta(
                target_signature,
                delta,
            )
            if donor_signature is None:
                continue
            for donor_formula in signature_to_formulas.get(
                donor_signature,
                [],
            ):
                if any(
                    candidates_by_id[donor_id]
                    for donor_id in formula_to_ids[donor_formula]
                ):
                    rules.append((operation, donor_formula))
        formula_rules_by_target[target_formula] = rules

    print(
        "[INFO] Formula-lattice neighbor index built: "
        f"unique_formulas={len(formula_to_ids)} "
        f"active_links={sum(len(v) for v in formula_rules_by_target.values())}",
        flush=True,
    )
    print("[INFO] Computing direct-peak and neutral-loss batch similarities lazily...", flush=True)
    spectral_similarity = SpectralSimilarityCache(selected_rows)

    output_rows = []
    source_counts: collections.Counter[str] = collections.Counter()
    originally_nonempty = 0
    originally_empty = 0
    own_top1_changed = 0
    own_top10_missing_rows = 0
    molecule_cache: dict[str, Chem.Mol | None] = {}
    transformation_cache: dict[tuple[str, str, str], Any] = {}
    start_time = time.perf_counter()

    for index, identifier in enumerate(order, 1):
        formula = str(selected_by_id[identifier]["true_formula"])
        nonempty = bool(candidates_by_id[identifier])
        originally_nonempty += int(nonempty)
        originally_empty += int(not nonempty)

        transfer_candidates = build_same_formula_transfer(
            target_id=identifier,
            target_formula=formula,
            formula_to_ids=formula_to_ids,
            candidates_by_id=candidates_by_id,
            spectral_similarity=spectral_similarity,
            top_k=args.top_k,
        )
        transformed_candidates = generate_formula_lattice_candidates(
            target_id=identifier,
            selected_by_id=selected_by_id,
            formula_to_ids=formula_to_ids,
            formula_rules_by_target=formula_rules_by_target,
            candidates_by_id=candidates_by_id,
            spectral_similarity=spectral_similarity,
            landmarks=landmarks,
            molecule_cache=molecule_cache,
            transformation_cache=transformation_cache,
        )
        final_candidates = merge_candidate_routes(
            original_nonempty=nonempty,
            original_candidates=candidates_by_id[identifier],
            transfer_candidates=transfer_candidates,
            transformed_candidates=transformed_candidates,
            top_k=args.top_k,
            lock_mode=args.lock_mode,
        )

        if nonempty:
            original_first = candidates_by_id[identifier][0]["smiles"]
            final_first = final_candidates[0]["smiles"] if final_candidates else None
            own_top1_changed += int(original_first != final_first)

            original_topk = {
                candidate["smiles"]
                for candidate in candidates_by_id[identifier][:args.top_k]
            }
            final_topk = {
                candidate["smiles"]
                for candidate in final_candidates[:args.top_k]
            }
            own_top10_missing_rows += int(
                not original_topk.issubset(final_topk)
            )

        for candidate in final_candidates:
            source_counts[candidate["prediction_source"]] += 1

        output_rows.append(
            {
                "identifier": identifier,
                "formula": formula,
                "status": "ok" if final_candidates else "no_candidates",
                "originally_nonempty": nonempty,
                "predictions": [
                    candidate["smiles"]
                    for candidate in final_candidates
                ],
                "candidates": final_candidates,
                "runtime_database_access": False,
                "external_structure_retrieval": False,
                "train_fold_structure_retrieval": False,
                "llm_used": False,
                "method": (
                    "current-batch same-formula spectral transfer + "
                    "formula-lattice chemical editing + "
                    f"{args.lock_mode} accuracy lock"
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
                f"[PREDICT] {index}/{len(order)} {identifier} "
                f"base={len(candidates_by_id[identifier])} "
                f"same_formula={len(transfer_candidates)} "
                f"transforms={len(transformed_candidates)} "
                f"final={len(final_candidates)} "
                f"rate={rate:.2f}/s eta={eta/60:.1f}m",
                flush=True,
            )

    output_path = Path(args.output)
    write_jsonl(output_path, output_rows)

    audit = {
        "num_rows": len(output_rows),
        "originally_nonempty_rows": originally_nonempty,
        "originally_empty_rows": originally_empty,
        "rows_with_final_candidates": sum(
            bool(row["predictions"])
            for row in output_rows
        ),
        "own_top1_changed_rows": own_top1_changed,
        "own_top10_missing_rows": own_top10_missing_rows,
        "lock_mode": args.lock_mode,
        "candidate_source_counts": dict(source_counts),
        "database_access": False,
        "external_candidate_files": [],
        "candidate_universe": (
            "Only MSNovelist candidates generated for spectra in the current "
            f"{len(order)}-sample batch, plus deterministic RDKit formula-delta edits."
        ),
        "truth_fields_read_by_predictor": [],
        "landmark_ranks": landmarks,
        "expected_rows": args.expected_rows,
        "spectral_pairs_computed": spectral_similarity.computed_pairs,
        "unique_formula_count": len(formula_to_ids),
        "formula_lattice_active_links": sum(
            len(value) for value in formula_rules_by_target.values()
        ),
        "molecule_cache_size": len(molecule_cache),
        "transformation_cache_size": len(transformation_cache),
        "elapsed_seconds": time.perf_counter() - start_time,
    }
    Path(args.audit).parent.mkdir(parents=True, exist_ok=True)
    Path(args.audit).write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if own_top1_changed:
        raise RuntimeError(
            f"Top-1 lock failed for {own_top1_changed} originally nonempty rows"
        )
    if args.lock_mode == "top10" and own_top10_missing_rows:
        raise RuntimeError(
            "Top-10 lock failed for "
            f"{own_top10_missing_rows} originally nonempty rows"
        )

    print("[OK] Predictions:", output_path)
    print("[OK] Audit:", args.audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
