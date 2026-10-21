#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

RDLogger.DisableLog("rdApp.error")
RDLogger.DisableLog("rdApp.warning")

MASS_SPEC_GYM_ID = re.compile(r"MassSpecGymID\d+")


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
                raise RuntimeError(f"{path}:{line_number}: row is not an object")
            rows.append(row)
    return rows


def optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def parse_formula(formula: str) -> dict[str, int]:
    parts = re.findall(r"([A-Z][a-z]?)(\d*)", formula.strip())
    if "".join(element + count for element, count in parts) != formula.strip():
        raise ValueError(f"Unsupported formula: {formula}")
    result: dict[str, int] = {}
    for element, count in parts:
        result[element] = result.get(element, 0) + int(count or "1")
    return result


def formula_equal(left: str, right: str) -> bool:
    try:
        return parse_formula(left) == parse_formula(right)
    except ValueError:
        return False


def choose_identifier(
    row: dict[str, str],
    selected_ids: set[str],
) -> str | None:
    """
    SIRIUS 6.3.7 may export mappingFeatureId values such as:

      872457286638793191_MassSpecGymID0399541_MassSpecGymID0399541

    Therefore exact equality is insufficient. We first try exact IDs, then
    extract MassSpecGymID tokens from the identifier-related columns, and
    finally scan all fields as a defensive fallback.
    """
    id_columns = (
        "mappingFeatureId",
        "compoundId",
        "alignedFeatureId",
        "featureId",
        "name",
    )

    # 1. Direct exact match.
    for key in id_columns:
        value = str(row.get(key) or "").strip()
        if value in selected_ids:
            return value

    # 2. Extract embedded MassSpecGym IDs from ID-related fields.
    matches: list[str] = []
    for key in id_columns:
        value = str(row.get(key) or "")
        for match in MASS_SPEC_GYM_ID.findall(value):
            if match in selected_ids:
                matches.append(match)

    unique = list(dict.fromkeys(matches))
    if len(unique) == 1:
        return unique[0]
    if len(unique) > 1:
        raise RuntimeError(
            "A SIRIUS row maps to multiple selected identifiers: "
            f"{unique}; mappingFeatureId={row.get('mappingFeatureId')!r}"
        )

    # 3. Defensive fallback: scan all fields.
    matches = []
    for value in row.values():
        for match in MASS_SPEC_GYM_ID.findall(str(value or "")):
            if match in selected_ids:
                matches.append(match)

    unique = list(dict.fromkeys(matches))
    if len(unique) == 1:
        return unique[0]
    if len(unique) > 1:
        raise RuntimeError(
            f"A SIRIUS row contains multiple selected identifiers: {unique}"
        )
    return None


def canonical_formula_exact(
    smiles: str,
    formula: str,
) -> tuple[str, str] | None:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    if len(Chem.GetMolFrags(molecule)) != 1:
        return None
    if Chem.GetFormalCharge(molecule) != 0:
        return None
    if any(atom.GetNumRadicalElectrons() for atom in molecule.GetAtoms()):
        return None

    actual_formula = rdMolDescriptors.CalcMolFormula(molecule)
    if not formula_equal(actual_formula, formula):
        return None

    canonical = Chem.MolToSmiles(
        molecule,
        canonical=True,
        isomericSmiles=False,
    )
    try:
        key2d = Chem.MolToInchiKey(molecule).split("-")[0]
    except Exception:
        key2d = canonical
    return canonical, key2d


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Parse SIRIUS denovo_structure_identifications_top-K.tsv and "
            "recover embedded MassSpecGym identifiers."
        )
    )
    parser.add_argument("--selected", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit", default=None)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--top-k", type=int, default=100)
    args = parser.parse_args()

    selected_rows = read_jsonl(Path(args.selected))
    if len(selected_rows) != args.expected_rows:
        raise RuntimeError(
            f"Expected {args.expected_rows} selected rows, "
            f"found {len(selected_rows)}"
        )

    selected_by_id = {
        str(row["identifier"]): row
        for row in selected_rows
    }
    selected_ids = set(selected_by_id)

    candidates_by_id: dict[str, list[dict[str, Any]]] = {
        identifier: []
        for identifier in selected_by_id
    }
    parse_invalid = {identifier: 0 for identifier in selected_by_id}
    formula_invalid = {identifier: 0 for identifier in selected_by_id}

    summary_path = Path(args.summary)
    if not summary_path.is_file():
        raise RuntimeError(f"Summary file does not exist: {summary_path}")

    data_rows = 0
    mapped_rows = 0
    unmapped_rows = 0
    empty_smiles_rows = 0
    accepted_rows = 0
    sample_unmapped_values: list[str] = []

    with summary_path.open(
        "r",
        encoding="utf-8",
        errors="replace",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = reader.fieldnames or []

        required_columns = {
            "structurePerIdRank",
            "molecularFormula",
            "smiles",
            "mappingFeatureId",
        }
        missing_columns = required_columns - set(fieldnames)
        if missing_columns:
            raise RuntimeError(
                "SIRIUS summary is missing required columns: "
                f"{sorted(missing_columns)}; actual={fieldnames}"
            )

        for tsv_row in reader:
            data_rows += 1
            identifier = choose_identifier(tsv_row, selected_ids)
            if identifier is None:
                unmapped_rows += 1
                if len(sample_unmapped_values) < 10:
                    sample_unmapped_values.append(
                        str(tsv_row.get("mappingFeatureId") or "")
                    )
                continue

            mapped_rows += 1
            smiles = str(tsv_row.get("smiles") or "").strip()
            if not smiles:
                empty_smiles_rows += 1
                continue

            target_formula = str(
                selected_by_id[identifier].get("true_formula")
                or selected_by_id[identifier].get("formula")
                or ""
            )
            if not target_formula:
                raise RuntimeError(
                    f"{identifier}: selected row has no true_formula/formula"
                )

            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                parse_invalid[identifier] += 1
                continue

            actual_formula = rdMolDescriptors.CalcMolFormula(molecule)
            if not formula_equal(actual_formula, target_formula):
                formula_invalid[identifier] += 1
                continue

            exact = canonical_formula_exact(smiles, target_formula)
            if exact is None:
                parse_invalid[identifier] += 1
                continue

            canonical, calculated_key2d = exact
            exported_key2d = str(tsv_row.get("InChIkey2D") or "").strip()
            key2d = exported_key2d or calculated_key2d

            rank = (
                optional_int(tsv_row.get("structurePerIdRank"))
                or len(candidates_by_id[identifier]) + 1
            )
            candidate = {
                "smiles": canonical,
                "inchikey2d": key2d,
                "source": "msnovelist_denovo",
                "rank": rank,
                "formula_rank": optional_int(tsv_row.get("formulaRank")),
                "generator_raw": optional_float(
                    tsv_row.get("CSI:FingerIDScore")
                ),
                "generator_score_name": "CSI:FingerIDScore",
                "sirius_scores": {
                    "CSI:FingerIDScore": optional_float(
                        tsv_row.get("CSI:FingerIDScore")
                    ),
                    "ModelScore": optional_float(
                        tsv_row.get("ModelScore")
                    ),
                    "ZodiacScore": optional_float(
                        tsv_row.get("ZodiacScore")
                    ),
                    "SiriusScoreNormalized": optional_float(
                        tsv_row.get("SiriusScoreNormalized")
                    ),
                    "SiriusScore": optional_float(
                        tsv_row.get("SiriusScore")
                    ),
                },
                "mapping_feature_id": tsv_row.get("mappingFeatureId"),
            }
            candidates_by_id[identifier].append(candidate)
            accepted_rows += 1

    if data_rows == 0:
        raise RuntimeError("SIRIUS summary contains no data rows")
    if mapped_rows == 0:
        raise RuntimeError(
            "Mapped zero SIRIUS rows to selected identifiers. "
            f"Sample mappingFeatureId values: {sample_unmapped_values}"
        )

    output_rows = []
    nonempty = 0
    empty = 0
    candidate_total = 0

    for index, selected_row in enumerate(selected_rows, 1):
        identifier = str(selected_row["identifier"])

        unique: dict[str, dict[str, Any]] = {}
        ordered = sorted(
            candidates_by_id[identifier],
            key=lambda row: (
                int(row["rank"]),
                row["smiles"],
            ),
        )
        for candidate in ordered:
            unique.setdefault(candidate["inchikey2d"], candidate)
            if len(unique) >= args.top_k:
                break

        candidates = list(unique.values())
        for rank, candidate in enumerate(candidates, 1):
            candidate["rank"] = rank

        if candidates:
            nonempty += 1
        else:
            empty += 1
        candidate_total += len(candidates)

        output_rows.append(
            {
                "identifier": identifier,
                "formula": str(
                    selected_row.get("true_formula")
                    or selected_row.get("formula")
                    or ""
                ),
                "true_smiles": selected_row.get("true_smiles"),
                "status": "ok" if candidates else "no_denovo_candidates",
                "num_candidates": len(candidates),
                "num_parse_invalid": parse_invalid[identifier],
                "num_formula_invalid": formula_invalid[identifier],
                "candidates": candidates,
            }
        )

        if index % 100 == 0 or index == len(selected_rows):
            print(
                f"[PARSE] {index}/{len(selected_rows)} "
                f"nonempty={nonempty} empty={empty}",
                flush=True,
            )

    # Hard guard against silently evaluating an empty ingestion.
    if nonempty == 0:
        raise RuntimeError(
            "Parsed zero nonempty candidate rows; refusing to write an "
            "all-empty baseline."
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    audit = {
        "selected_rows": len(selected_rows),
        "tsv_data_rows": data_rows,
        "mapped_tsv_rows": mapped_rows,
        "unmapped_tsv_rows": unmapped_rows,
        "empty_smiles_rows": empty_smiles_rows,
        "accepted_formula_exact_tsv_rows": accepted_rows,
        "nonempty_samples": nonempty,
        "empty_samples": empty,
        "candidate_total_after_dedup_topk": candidate_total,
        "average_candidates_per_sample": (
            candidate_total / len(selected_rows)
        ),
        "parse_invalid_total": sum(parse_invalid.values()),
        "formula_invalid_total": sum(formula_invalid.values()),
        "sample_unmapped_mappingFeatureId": sample_unmapped_values,
    }
    audit_path = (
        Path(args.audit)
        if args.audit
        else output_path.with_suffix(".audit.json")
    )
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print("[OK] Parsed candidates:", output_path)
    print("[OK] Parse audit:", audit_path)


if __name__ == "__main__":
    main()
