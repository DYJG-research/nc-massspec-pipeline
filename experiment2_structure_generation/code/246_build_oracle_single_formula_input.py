#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path


def load_records(path):
    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()

    if not text:
        return []

    try:
        obj = json.loads(text)

        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]

        if isinstance(obj, dict):
            for key in (
                "data",
                "items",
                "records",
                "spectra",
                "examples",
            ):
                value = obj.get(key)
                if isinstance(value, list):
                    return [
                        x for x in value
                        if isinstance(x, dict)
                    ]

            return [obj]

    except json.JSONDecodeError:
        pass

    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)
            except Exception as exc:
                raise RuntimeError(
                    f"{path}:{line_no}: {exc}"
                ) from exc

            if isinstance(row, dict):
                rows.append(row)

    return rows


def load_module(path):
    path = Path(path).resolve()

    spec = importlib.util.spec_from_file_location(
        "oracle_formula_chemistry",
        str(path),
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Cannot import chemistry module: {path}"
        )

    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    return mod


def reference_formula(row, chem):
    molecule = row.get("molecule") or {}
    metadata = row.get("metadata") or {}

    values = [
        molecule.get("formula"),
        row.get("true_formula"),
        row.get("formula"),
        metadata.get("formula"),
    ]

    for value in values:
        if not value:
            continue

        formula = chem.normalize_formula(
            str(value)
        )

        if formula:
            return formula

    return None


def normalize_peaks(peaks):
    output = []

    for p in peaks or []:
        if (
            not isinstance(p, (list, tuple))
            or len(p) < 2
        ):
            continue

        try:
            mz = float(p[0])
            intensity = float(p[1])
        except Exception:
            continue

        if (
            math.isfinite(mz)
            and math.isfinite(intensity)
            and mz > 0
            and intensity >= 0
        ):
            output.append(
                [mz, intensity]
            )

    return output


def percentile(values, q):
    if not values:
        return None

    values = sorted(values)

    if len(values) == 1:
        return values[0]

    pos = (len(values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)

    if lo == hi:
        return values[lo]

    w = pos - lo

    return (
        values[lo] * (1.0 - w)
        + values[hi] * w
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--massspecgym",
        required=True,
    )

    ap.add_argument(
        "--top5-branches",
        required=True,
    )

    ap.add_argument(
        "--chemistry",
        required=True,
    )

    ap.add_argument(
        "--output",
        required=True,
    )

    ap.add_argument(
        "--audit",
        required=True,
    )

    ap.add_argument(
        "--expected-samples",
        type=int,
        default=17556,
    )

    args = ap.parse_args()

    chem = load_module(
        args.chemistry
    )

    # ---------------------------------------------------------
    # 1. Reuse EXACT observable spectrum information from
    #    the existing Top-5 branch manifest.
    # ---------------------------------------------------------

    branch_rows = load_records(
        args.top5_branches
    )

    sample_obs = {}
    sample_order = []

    for row in branch_rows:

        sid = str(
            row["sample_id"]
        )

        precursor_mz = float(
            row["precursor_mz"]
        )

        polarity = chem.normalize_polarity(
            row["polarity"]
        )

        if polarity not in (
            "positive",
            "negative",
        ):
            raise RuntimeError(
                f"Invalid polarity for {sid}: "
                f"{row.get('polarity')}"
            )

        peaks = normalize_peaks(
            row.get("peaks")
        )

        obs = {
            "precursor_mz":
                precursor_mz,
            "polarity":
                polarity,
            "peaks":
                peaks,
        }

        if sid not in sample_obs:
            sample_obs[sid] = obs
            sample_order.append(sid)

        else:
            old = sample_obs[sid]

            if (
                old["precursor_mz"]
                != obs["precursor_mz"]
                or old["polarity"]
                != obs["polarity"]
                or old["peaks"]
                != obs["peaks"]
            ):
                raise RuntimeError(
                    f"Inconsistent observable data "
                    f"between formula branches: {sid}"
                )

    if len(sample_order) != args.expected_samples:
        raise RuntimeError(
            f"Expected {args.expected_samples} samples "
            f"in Top-5 manifest, found "
            f"{len(sample_order)}"
        )

    # ---------------------------------------------------------
    # 2. Read ONLY the reference molecular formula.
    #    Do not read GT SMILES or GT adduct.
    # ---------------------------------------------------------

    mass_rows = load_records(
        args.massspecgym
    )

    truth_formula = {}

    for row in mass_rows:

        sid = str(
            row.get("identifier")
            or row.get("id")
            or ""
        )

        if not sid:
            continue

        formula = reference_formula(
            row,
            chem,
        )

        if formula:
            truth_formula[sid] = formula

    missing_formula = [
        sid
        for sid in sample_order
        if sid not in truth_formula
    ]

    if missing_formula:
        raise RuntimeError(
            "Missing reference formula for "
            f"{len(missing_formula)} samples. "
            f"Examples: {missing_formula[:10]}"
        )

    # ---------------------------------------------------------
    # 3. Construct exactly ONE oracle formula hypothesis.
    #
    # GT adduct is NOT used.
    # Latent adduct = allowed adduct with minimum |ppm|.
    # ---------------------------------------------------------

    outputs = []

    adduct_counts = Counter()
    abs_ppms = []
    gt_field_names_written = set()

    for sid in sample_order:

        obs = sample_obs[sid]

        precursor_mz = float(
            obs["precursor_mz"]
        )

        polarity = obs["polarity"]

        formula = truth_formula[sid]

        counts = chem.parse_formula(
            formula
        )

        if not counts:
            raise RuntimeError(
                f"Cannot parse reference formula "
                f"{sid}: {formula}"
            )

        neutral_mass = chem.exact_mass(
            counts
        )

        dbe = chem.calc_dbe(
            counts
        )

        allowed = chem.ADDUCTS.get(
            polarity
        )

        if not allowed:
            raise RuntimeError(
                f"No allowed adducts for "
                f"{sid}/{polarity}"
            )

        candidates = []

        for adduct in allowed:

            calculated_mz = (
                neutral_mass
                + float(adduct.mass_shift)
            )

            ppm_error = (
                (calculated_mz - precursor_mz)
                / precursor_mz
                * 1_000_000.0
            )

            candidates.append(
                (
                    abs(ppm_error),
                    int(adduct.priority),
                    str(adduct.name),
                    float(ppm_error),
                )
            )

        candidates.sort(
            key=lambda x: (
                x[0],
                x[1],
                x[2],
            )
        )

        (
            abs_ppm,
            _priority,
            inferred_adduct,
            ppm_error,
        ) = candidates[0]

        adduct_counts[
            inferred_adduct
        ] += 1

        abs_ppms.append(
            abs_ppm
        )

        row = {
            "schema_version":
                "oracle-single-formula-v1",

            "sample_id":
                sid,

            "precursor_mz":
                precursor_mz,

            "polarity":
                polarity,

            "peaks":
                obs["peaks"],

            "formula_hypotheses": [
                {
                    "rank":
                        1,

                    "formula":
                        formula,

                    "adduct":
                        inferred_adduct,

                    # Oracle has no learned formula score.
                    # Keep it neutral and identical
                    # for every sample.
                    "score":
                        0.0,

                    # Provenance only.
                    # Formula-Lattice does not use
                    # confidence for ranking.
                    "confidence":
                        1.0,

                    "ppm_error":
                        ppm_error,

                    "dbe":
                        dbe,
                }
            ],
        }

        forbidden = {
            "true_formula",
            "true_adduct",
            "true_smiles",
            "smiles",
            "inchikey",
            "gold",
            "gold_formula",
            "gold_smiles",
        }

        gt_field_names_written.update(
            set(row) & forbidden
        )

        outputs.append(
            row
        )

    # ---------------------------------------------------------
    # 4. Hard assertions
    # ---------------------------------------------------------

    if len(outputs) != args.expected_samples:
        raise RuntimeError(
            f"Output rows = {len(outputs)}, "
            f"expected {args.expected_samples}"
        )

    if len({
        x["sample_id"]
        for x in outputs
    }) != args.expected_samples:
        raise RuntimeError(
            "Duplicate sample_id in output"
        )

    for row in outputs:
        hs = row["formula_hypotheses"]

        if len(hs) != 1:
            raise RuntimeError(
                f"{row['sample_id']}: "
                f"formula count != 1"
            )

        if int(hs[0]["rank"]) != 1:
            raise RuntimeError(
                f"{row['sample_id']}: "
                f"oracle formula rank != 1"
            )

    # ---------------------------------------------------------
    # 5. Write output
    # ---------------------------------------------------------

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for row in outputs:

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )

    over10 = sum(
        x > 10.0
        for x in abs_ppms
    )

    audit = {
        "method":
            "Oracle Formula only",

        "oracle_information":
            "one reference neutral molecular formula per spectrum",

        "num_samples":
            len(outputs),

        "unique_sample_ids":
            len({
                x["sample_id"]
                for x in outputs
            }),

        "formula_hypotheses_per_sample":
            1,

        "formula_rank":
            1,

        "gt_smiles_used":
            False,

        "gt_inchikey_used":
            False,

        "annotated_adduct_used":
            False,

        "adduct_policy":
            "minimum absolute precursor ppm among the same allowed latent-adduct set",

        "adduct_counts":
            dict(adduct_counts),

        "abs_ppm": {
            "mean":
                statistics.fmean(
                    abs_ppms
                )
                if abs_ppms
                else None,

            "median":
                statistics.median(
                    abs_ppms
                )
                if abs_ppms
                else None,

            "p95":
                percentile(
                    abs_ppms,
                    0.95,
                ),

            "max":
                max(abs_ppms)
                if abs_ppms
                else None,

            "le_10ppm":
                len(abs_ppms) - over10,

            "gt_10ppm":
                over10,
        },

        "forbidden_gt_fields_written":
            sorted(
                gt_field_names_written
            ),

        "observable_data_source":
            str(
                Path(
                    args.top5_branches
                ).resolve()
            ),

        "reference_formula_source":
            str(
                Path(
                    args.massspecgym
                ).resolve()
            ),
    }

    audit_path = Path(
        args.audit
    )

    audit_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit_path.write_text(
        json.dumps(
            audit,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "=" * 72
    )
    print(
        "ORACLE SINGLE-FORMULA INPUT"
    )
    print(
        "=" * 72
    )

    print(
        "Samples              :",
        len(outputs),
    )

    print(
        "Formula/sample       : 1"
    )

    print(
        "GT SMILES used       : NO"
    )

    print(
        "GT adduct used       : NO"
    )

    print(
        "Abs ppm <= 10        :",
        len(abs_ppms) - over10,
    )

    print(
        "Abs ppm > 10         :",
        over10,
    )

    print(
        "Adduct distribution  :",
        dict(adduct_counts),
    )

    print(
        "Output               :",
        output_path.resolve(),
    )

    print(
        "Audit                :",
        audit_path.resolve(),
    )

    print(
        "=" * 72
    )


if __name__ == "__main__":
    main()
