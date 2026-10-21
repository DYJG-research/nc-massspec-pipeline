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


BRANCH_SUFFIX_PATTERN = re.compile(
    r"__F\d{2}__[0-9a-f]{10}"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:

    rows = []

    with path.open(
        "r",
        encoding="utf-8"
    ) as handle:

        for line_number, line in enumerate(
            handle,
            1
        ):

            if not line.strip():
                continue

            try:
                row = json.loads(line)

            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"{path}:{line_number}: invalid JSON"
                ) from exc

            if not isinstance(row, dict):
                raise RuntimeError(
                    f"{path}:{line_number}: row is not object"
                )

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


def parse_formula(
    formula: str
) -> dict[str, int]:

    formula = formula.strip()

    parts = re.findall(
        r"([A-Z][a-z]?)(\d*)",
        formula
    )

    reconstructed = "".join(
        element + count
        for element, count in parts
    )

    if reconstructed != formula:
        raise ValueError(
            f"Unsupported neutral formula: {formula}"
        )

    result: dict[str, int] = {}

    for element, count in parts:

        result[element] = (
            result.get(element, 0)
            + int(count or "1")
        )

    return result


def formula_equal(
    left: str,
    right: str
) -> bool:

    try:
        return (
            parse_formula(left)
            ==
            parse_formula(right)
        )

    except ValueError:
        return False


def build_branch_suffix_map(
    branch_ids: set[str]
) -> dict[str, str]:

    mapping = {}

    for branch_id in branch_ids:

        match = BRANCH_SUFFIX_PATTERN.search(
            branch_id
        )

        if match is None:
            raise RuntimeError(
                "branch_id does not contain expected "
                f"rank/hash suffix: {branch_id}"
            )

        suffix = match.group(0)

        if (
            suffix in mapping
            and mapping[suffix] != branch_id
        ):
            raise RuntimeError(
                f"Non-unique branch suffix: {suffix}"
            )

        mapping[suffix] = branch_id

    return mapping


def choose_branch_id(
    row: dict[str, str],
    branch_ids: set[str],
    suffix_map: dict[str, str],
) -> str | None:

    id_columns = (
        "mappingFeatureId",
        "compoundId",
        "alignedFeatureId",
        "featureId",
        "name",
    )

    # 1. Direct exact equality.
    for key in id_columns:

        value = str(
            row.get(key) or ""
        ).strip()

        if value in branch_ids:
            return value

    # 2. SIRIUS may wrap feature IDs with prefixes/suffixes.
    #    Our rank/hash suffix remains stable and unique.
    matches = []

    for key in id_columns:

        value = str(
            row.get(key) or ""
        )

        for suffix in BRANCH_SUFFIX_PATTERN.findall(
            value
        ):

            branch_id = suffix_map.get(
                suffix
            )

            if branch_id is not None:
                matches.append(branch_id)

    unique = list(
        dict.fromkeys(matches)
    )

    if len(unique) == 1:
        return unique[0]

    if len(unique) > 1:
        raise RuntimeError(
            "SIRIUS row maps to multiple branches: "
            f"{unique}; "
            f"mappingFeatureId="
            f"{row.get('mappingFeatureId')!r}"
        )

    # 3. Defensive fallback over every TSV field.
    matches = []

    for value in row.values():

        text = str(value or "")

        for suffix in BRANCH_SUFFIX_PATTERN.findall(
            text
        ):

            branch_id = suffix_map.get(
                suffix
            )

            if branch_id is not None:
                matches.append(branch_id)

    unique = list(
        dict.fromkeys(matches)
    )

    if len(unique) == 1:
        return unique[0]

    if len(unique) > 1:
        raise RuntimeError(
            "SIRIUS row contains multiple branch IDs: "
            f"{unique}"
        )

    return None


def canonical_formula_exact(
    smiles: str,
    target_formula: str,
) -> tuple[str, str] | None:

    molecule = Chem.MolFromSmiles(
        smiles
    )

    if molecule is None:
        return None

    # Single connected molecule only.
    if len(
        Chem.GetMolFrags(molecule)
    ) != 1:
        return None

    # Our formula hypotheses are neutral molecular formulas.
    if Chem.GetFormalCharge(
        molecule
    ) != 0:
        return None

    # Reject radicals.
    if any(
        atom.GetNumRadicalElectrons()
        for atom in molecule.GetAtoms()
    ):
        return None

    actual_formula = (
        rdMolDescriptors.CalcMolFormula(
            molecule
        )
    )

    if not formula_equal(
        actual_formula,
        target_formula
    ):
        return None

    canonical = Chem.MolToSmiles(
        molecule,
        canonical=True,
        isomericSmiles=False,
    )

    try:
        key2d = (
            Chem.MolToInchiKey(
                molecule
            ).split("-")[0]
        )

    except Exception:
        key2d = canonical

    return canonical, key2d


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Parse SIRIUS/MSNovelist outputs for "
            "multi-formula branches."
        )
    )

    parser.add_argument(
        "--branches",
        required=True
    )

    parser.add_argument(
        "--summary",
        required=True
    )

    parser.add_argument(
        "--output",
        required=True
    )

    parser.add_argument(
        "--audit",
        required=True
    )

    parser.add_argument(
        "--expected-branches",
        type=int,
        required=True
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=100
    )

    parser.add_argument(
        "--require-nonempty",
        action="store_true"
    )

    args = parser.parse_args()

    branch_rows = read_jsonl(
        Path(args.branches)
    )

    if (
        len(branch_rows)
        != args.expected_branches
    ):
        raise RuntimeError(
            f"Expected {args.expected_branches} branches, "
            f"found {len(branch_rows)}"
        )

    branch_by_id = {}

    for row in branch_rows:

        branch_id = str(
            row["branch_id"]
        )

        if branch_id in branch_by_id:
            raise RuntimeError(
                f"Duplicate branch_id: {branch_id}"
            )

        branch_by_id[branch_id] = row

    branch_ids = set(
        branch_by_id
    )

    suffix_map = build_branch_suffix_map(
        branch_ids
    )

    candidates_by_branch = {
        branch_id: []
        for branch_id in branch_ids
    }

    parse_invalid = {
        branch_id: 0
        for branch_id in branch_ids
    }

    formula_invalid = {
        branch_id: 0
        for branch_id in branch_ids
    }

    summary_path = Path(
        args.summary
    )

    if not summary_path.is_file():
        raise RuntimeError(
            f"Summary does not exist: {summary_path}"
        )

    data_rows = 0
    mapped_rows = 0
    unmapped_rows = 0
    empty_smiles_rows = 0
    accepted_rows = 0

    sample_unmapped_values = []

    with summary_path.open(
        "r",
        encoding="utf-8",
        errors="replace",
        newline=""
    ) as handle:

        reader = csv.DictReader(
            handle,
            delimiter="\t"
        )

        fieldnames = (
            reader.fieldnames or []
        )

        required_columns = {
            "structurePerIdRank",
            "molecularFormula",
            "smiles",
            "mappingFeatureId",
        }

        missing_columns = (
            required_columns
            - set(fieldnames)
        )

        if missing_columns:
            raise RuntimeError(
                "SIRIUS summary missing columns: "
                f"{sorted(missing_columns)}; "
                f"actual={fieldnames}"
            )

        for tsv_row in reader:

            data_rows += 1

            branch_id = choose_branch_id(
                tsv_row,
                branch_ids,
                suffix_map
            )

            if branch_id is None:

                unmapped_rows += 1

                if len(
                    sample_unmapped_values
                ) < 20:

                    sample_unmapped_values.append(
                        str(
                            tsv_row.get(
                                "mappingFeatureId"
                            )
                            or ""
                        )
                    )

                continue

            mapped_rows += 1

            smiles = str(
                tsv_row.get("smiles")
                or ""
            ).strip()

            if not smiles:

                empty_smiles_rows += 1
                continue

            branch = branch_by_id[
                branch_id
            ]

            target_formula = str(
                branch["formula"]
            )

            molecule = Chem.MolFromSmiles(
                smiles
            )

            if molecule is None:

                parse_invalid[
                    branch_id
                ] += 1

                continue

            actual_formula = (
                rdMolDescriptors.CalcMolFormula(
                    molecule
                )
            )

            if not formula_equal(
                actual_formula,
                target_formula
            ):

                formula_invalid[
                    branch_id
                ] += 1

                continue

            exact = canonical_formula_exact(
                smiles,
                target_formula
            )

            if exact is None:

                parse_invalid[
                    branch_id
                ] += 1

                continue

            canonical, calculated_key2d = exact

            exported_key2d = str(
                tsv_row.get(
                    "InChIkey2D"
                )
                or ""
            ).strip()

            key2d = (
                exported_key2d
                or calculated_key2d
            )

            rank = (
                optional_int(
                    tsv_row.get(
                        "structurePerIdRank"
                    )
                )
                or (
                    len(
                        candidates_by_branch[
                            branch_id
                        ]
                    )
                    + 1
                )
            )

            candidate = {
                "smiles":
                    canonical,

                "inchikey2d":
                    key2d,

                "source":
                    "sirius_msnovelist",

                "rank":
                    rank,

                "sirius_formula_rank":
                    optional_int(
                        tsv_row.get(
                            "formulaRank"
                        )
                    ),

                "generator_raw":
                    optional_float(
                        tsv_row.get(
                            "CSI:FingerIDScore"
                        )
                    ),

                "generator_score_name":
                    "CSI:FingerIDScore",

                "sirius_scores": {
                    "CSI:FingerIDScore":
                        optional_float(
                            tsv_row.get(
                                "CSI:FingerIDScore"
                            )
                        ),

                    "ModelScore":
                        optional_float(
                            tsv_row.get(
                                "ModelScore"
                            )
                        ),

                    "ZodiacScore":
                        optional_float(
                            tsv_row.get(
                                "ZodiacScore"
                            )
                        ),

                    "SiriusScoreNormalized":
                        optional_float(
                            tsv_row.get(
                                "SiriusScoreNormalized"
                            )
                        ),

                    "SiriusScore":
                        optional_float(
                            tsv_row.get(
                                "SiriusScore"
                            )
                        ),
                },

                "mapping_feature_id":
                    tsv_row.get(
                        "mappingFeatureId"
                    ),
            }

            candidates_by_branch[
                branch_id
            ].append(
                candidate
            )

            accepted_rows += 1

    output_rows = []

    nonempty = 0
    empty = 0
    candidate_total = 0

    for branch in branch_rows:

        branch_id = str(
            branch["branch_id"]
        )

        unique = {}

        ordered = sorted(
            candidates_by_branch[
                branch_id
            ],
            key=lambda row: (
                int(row["rank"]),
                row["smiles"]
            )
        )

        for candidate in ordered:

            unique.setdefault(
                candidate["inchikey2d"],
                candidate
            )

            if len(unique) >= args.top_k:
                break

        candidates = list(
            unique.values()
        )

        for rank, candidate in enumerate(
            candidates,
            1
        ):
            candidate["rank"] = rank

        if candidates:
            nonempty += 1
        else:
            empty += 1

        candidate_total += len(
            candidates
        )

        output_rows.append(
            {
                "schema_version":
                    "multiformula-sirius-candidates-v1",

                "sample_id":
                    branch["sample_id"],

                "branch_id":
                    branch_id,

                "formula_rank":
                    int(
                        branch["formula_rank"]
                    ),

                "formula":
                    str(
                        branch["formula"]
                    ),

                "adduct":
                    branch.get(
                        "adduct"
                    ),

                "formula_score":
                    branch.get(
                        "formula_score"
                    ),

                "formula_confidence":
                    branch.get(
                        "formula_confidence"
                    ),

                "formula_ppm_error":
                    branch.get(
                        "formula_ppm_error"
                    ),

                "formula_dbe":
                    branch.get(
                        "formula_dbe"
                    ),

                "status":
                    (
                        "ok"
                        if candidates
                        else "no_denovo_candidates"
                    ),

                "num_candidates":
                    len(candidates),

                "num_parse_invalid":
                    parse_invalid[
                        branch_id
                    ],

                "num_formula_invalid":
                    formula_invalid[
                        branch_id
                    ],

                "candidates":
                    candidates,
            }
        )

    if (
        args.require_nonempty
        and nonempty == 0
    ):
        raise RuntimeError(
            "Parsed zero nonempty formula branches."
        )

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with output_path.open(
        "w",
        encoding="utf-8"
    ) as handle:

        for row in output_rows:

            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False
                )
                + "\n"
            )

    audit = {
        "branches":
            len(branch_rows),

        "tsv_data_rows":
            data_rows,

        "mapped_tsv_rows":
            mapped_rows,

        "unmapped_tsv_rows":
            unmapped_rows,

        "empty_smiles_rows":
            empty_smiles_rows,

        "accepted_formula_exact_tsv_rows":
            accepted_rows,

        "nonempty_branches":
            nonempty,

        "empty_branches":
            empty,

        "branch_coverage":
            (
                nonempty / len(branch_rows)
                if branch_rows
                else 0.0
            ),

        "candidate_total_after_dedup_topk":
            candidate_total,

        "average_candidates_per_branch":
            (
                candidate_total
                / len(branch_rows)
                if branch_rows
                else 0.0
            ),

        "parse_invalid_total":
            sum(
                parse_invalid.values()
            ),

        "formula_invalid_total":
            sum(
                formula_invalid.values()
            ),

        "sample_unmapped_mappingFeatureId":
            sample_unmapped_values,
    }

    audit_path = Path(
        args.audit
    )

    audit_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    audit_path.write_text(
        json.dumps(
            audit,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print(
        json.dumps(
            audit,
            ensure_ascii=False,
            indent=2
        )
    )

    print(
        "[OK] Branch candidates:",
        output_path
    )

    print(
        "[OK] Audit:",
        audit_path
    )


if __name__ == "__main__":
    main()
