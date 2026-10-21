#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build candidate-level datasets for a Broad-to-Compact XGBoost pre-ranker.

Default practical broad pool:
    V3 broad Top50
  ∪ OLD full Top400
  ∪ Direct Top5
  ∪ V3-broad-F Top15
  -> neutral-formula dedup

Each spectrum is one ranking group. Each candidate is one row.

Training:
- label = 1 only for the reference neutral formula, otherwise 0.
- By default, train groups with no gold candidate are dropped because they
  provide no within-group positive ranking signal.

Validation:
- use --no-drop-no-gold so that groups whose gold is absent remain in the
  denominator when Recall@K is computed.

Features are label-blind and available at inference:
- precursor/candidate mass evidence
- DBE and elemental composition
- V3/OLD/Direct/F global source membership and rank
- independent generator support
- simple spectrum summary features
- latent BestAdduct one-hot

No SIRIUS. No external database. No gold-dependent candidate generation.

Storage is compact:
- X float32 binary
- y uint8 binary
- group sizes int32 binary
- gzip JSONL metadata with candidate formulas per spectrum
- audit JSONL for resume

The script is multiprocessing-enabled and supports --resume.
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import math
import multiprocessing as mp
import os
import random
import re
import signal
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# =============================================================================
# Generic helpers
# =============================================================================

ELEMENTS = ["C", "H", "N", "O", "P", "S", "F", "Cl", "Br", "I", "Si"]
HALOGENS = ["F", "Cl", "Br", "I"]


def import_module_from_file(name: str, path: str):
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(p)
    spec = importlib.util.spec_from_file_location(name, str(p))
    if spec is None or spec.loader is None:
        raise ImportError(p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def canonical_adduct(x: Any) -> str:
    if x is None:
        return ""
    return re.sub(r"\s+", "", str(x).strip())


def get_nested(d: Dict[str, Any], path: Sequence[str], default=None):
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def get_true_adduct(item: Dict[str, Any]) -> str:
    for x in (
        get_nested(item, ["metadata", "adduct"]),
        item.get("true_adduct"),
        item.get("adduct"),
    ):
        if x:
            return canonical_adduct(x)
    return ""


def ordered_union(base, *lists: Sequence[str]) -> List[str]:
    out, seen = [], set()
    for values in lists:
        for formula in values:
            key = base.formula_key(formula)
            if key is None or key in seen:
                continue
            seen.add(key)
            out.append(formula)
    return out


def contains_element(base, formula: str, element: str) -> bool:
    counts = base.parse_formula(formula)
    return bool(counts) and counts.get(element, 0) > 0


def rank_map(base, formulas: Sequence[str]) -> Dict[Any, int]:
    out = {}
    for rank, formula in enumerate(formulas, 1):
        key = base.formula_key(formula)
        if key is not None and key not in out:
            out[key] = rank
    return out


def load_completed_audit(path: Path) -> set[int]:
    completed = set()
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            completed.add(int(row["fold_idx"]))
    return completed


def count_gzip_lines(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    xs = sorted(float(x) for x in values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


# =============================================================================
# Exact V3 current + broad
# =============================================================================

def build_v3_current_and_broad(base, pmz, polarity, frags, limits, args, signatures):
    original, expanded = [], []

    for adduct in base.ADDUCTS[polarity]:
        neutral = pmz - adduct.mass_shift
        if neutral <= 0:
            continue

        cr = base.carbon_range_v3(neutral, frags)

        for c, mass, d in base.enumerate_chno(
            neutral, args.v3_ppm, limits, cr
        ):
            original.append(
                base.make_candidate(c, mass, d, pmz, adduct, True)
            )

        for c, mass, d in base.enumerate_legacy_8(
            neutral, args.v3_ppm, limits, cr
        ):
            expanded.append(
                base.make_candidate(c, mass, d, pmz, adduct, True)
            )

        for sig in signatures:
            for c, mass, d in base.enumerate_extension(
                neutral, args.v3_ppm, sig, limits, cr
            ):
                expanded.append(
                    base.make_candidate(c, mass, d, pmz, adduct, True)
                )

    strict = [c for c in expanded if c.rule_valid]

    of, ob = base.unique_best(
        original,
        lambda c: base.original_key(c, polarity, True),
    )
    rf, rb = base.unique_best(strict, base.rule_key)
    pf, pb = base.unique_best(
        strict,
        lambda c: base.prior_key(c, True),
    )

    branches = {
        "original": (of, ob),
        "rule": (rf, rb),
        "prior": (pf, pb),
    }

    current = base.union_ranked(
        branches,
        args.v3_source_top,
        True,
    )
    broad = base.union_ranked(
        branches,
        0,
        True,
    )

    return current, broad


# =============================================================================
# Feature construction
# =============================================================================

def safe_ratio(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def spectrum_summary(
    precursor_mz: float,
    mzs: Sequence[float],
    intensities: Sequence[float],
) -> Dict[str, float]:
    if not mzs:
        return {
            "spec_num_peaks": 0.0,
            "spec_base_mz_ratio": 0.0,
            "spec_weighted_mz_ratio": 0.0,
            "spec_high_mz_fraction": 0.0,
            "spec_top5_intensity_fraction": 0.0,
        }

    mz_arr = np.asarray(mzs, dtype=np.float64)
    int_arr = np.asarray(intensities, dtype=np.float64)

    if int_arr.size != mz_arr.size or int_arr.size == 0:
        int_arr = np.ones_like(mz_arr)

    int_arr = np.maximum(int_arr, 0.0)
    total_i = float(int_arr.sum())
    if total_i <= 0:
        int_arr = np.ones_like(mz_arr)
        total_i = float(int_arr.sum())

    base_idx = int(np.argmax(int_arr))
    base_mz = float(mz_arr[base_idx])
    weighted_mz = float((mz_arr * int_arr).sum() / total_i)

    high_cut = 0.5 * precursor_mz
    high_fraction = float(int_arr[mz_arr >= high_cut].sum() / total_i)

    topk = np.sort(int_arr)[-min(5, len(int_arr)):]
    top5_fraction = float(topk.sum() / total_i)

    return {
        "spec_num_peaks": float(len(mz_arr)),
        "spec_base_mz_ratio": safe_ratio(base_mz, precursor_mz),
        "spec_weighted_mz_ratio": safe_ratio(weighted_mz, precursor_mz),
        "spec_high_mz_fraction": high_fraction,
        "spec_top5_intensity_fraction": top5_fraction,
    }


def build_feature_names(adduct_names: Sequence[str]) -> List[str]:
    names = [
        "abs_ppm",
        "signed_ppm",
        "dbe",
        "precursor_mz",
        "neutral_mass",
        "mass_defect",
    ]

    names.extend([f"elem_{e}" for e in ELEMENTS])

    names.extend([
        "total_atoms",
        "heavy_atoms",
        "hetero_atoms",
        "halogen_atoms",
        "rare_element_types",
        "H_over_C",
        "N_over_C",
        "O_over_C",
        "hetero_over_C",
        "DBE_over_C",

        "v3_present",
        "v3_rank",
        "v3_rank_norm",
        "v3_rr",

        "old_present",
        "old_rank",
        "old_rank_norm",
        "old_rr",

        "direct_present",
        "direct_rank",
        "direct_rank_norm",
        "direct_rr",

        "f_present",
        "f_rank",
        "f_rank_norm",
        "f_rr",

        "support",
        "best_rank_norm",
        "rr_sum",
        "is_chno_only",
        "polarity_positive",

        "spec_num_peaks",
        "spec_base_mz_ratio",
        "spec_weighted_mz_ratio",
        "spec_high_mz_fraction",
        "spec_top5_intensity_fraction",
    ])

    names.extend([f"adduct_{a}" for a in adduct_names])
    return names


def candidate_features(
    base,
    builder,
    formula: str,
    precursor_mz: float,
    polarity: str,
    spec: Dict[str, float],
    v3_rank: Optional[int],
    old_rank: Optional[int],
    direct_rank: Optional[int],
    f_rank: Optional[int],
    v3_top: int,
    old_top: int,
    direct_top: int,
    f_top: int,
    adduct_names: Sequence[str],
) -> np.ndarray:
    counts = base.parse_formula(formula)
    if not counts:
        raise ValueError(f"Cannot parse formula {formula!r}")

    neutral_mass = float(base.exact_mass(counts))
    chemistry = builder.best_mass_matched_adduct(
        base,
        formula,
        precursor_mz,
        polarity,
    )

    ppm = chemistry.get("ppm_error")
    ppm = float(ppm) if ppm is not None else math.nan

    dbe = chemistry.get("dbe")
    dbe = float(dbe) if dbe is not None else math.nan

    element_counts = {
        e: int(counts.get(e, 0))
        for e in ELEMENTS
    }

    total_atoms = sum(element_counts.values())
    heavy_atoms = total_atoms - element_counts["H"]
    hetero_atoms = sum(
        element_counts[e]
        for e in ELEMENTS
        if e not in ("C", "H")
    )
    halogen_atoms = sum(
        element_counts[e]
        for e in HALOGENS
    )
    rare_types = sum(
        int(element_counts[e] > 0)
        for e in ("F", "I", "Si")
    )

    c = element_counts["C"]
    h = element_counts["H"]
    n = element_counts["N"]
    o = element_counts["O"]

    v3_present = int(v3_rank is not None)
    old_present = int(old_rank is not None)
    direct_present = int(direct_rank is not None)
    f_present = int(f_rank is not None)

    # V3 and F are one generator family for Support.
    support = (
        int(v3_present or f_present)
        + old_present
        + direct_present
    )

    def rank_numeric(rank: Optional[int]) -> float:
        return float(rank) if rank is not None else math.nan

    def rank_norm(rank: Optional[int], cutoff: int) -> float:
        return (
            float(rank) / float(cutoff)
            if rank is not None and cutoff > 0
            else math.nan
        )

    def rr(rank: Optional[int]) -> float:
        return 1.0 / float(rank) if rank is not None and rank > 0 else 0.0

    norm_ranks = [
        x for x in (
            rank_norm(v3_rank, v3_top),
            rank_norm(old_rank, old_top),
            rank_norm(direct_rank, direct_top),
            rank_norm(f_rank, f_top),
        )
        if not math.isnan(x)
    ]

    best_rank_norm = min(norm_ranks) if norm_ranks else 1.5
    rr_sum = (
        rr(v3_rank)
        + rr(old_rank)
        + rr(direct_rank)
        + rr(f_rank)
    )

    is_chno_only = int(
        all(
            element_counts[e] == 0
            for e in ("P", "S", "F", "Cl", "Br", "I", "Si")
        )
    )

    best_adduct = canonical_adduct(
        chemistry.get("best_adduct", "")
    )

    values: List[float] = [
        abs(ppm) if not math.isnan(ppm) else math.nan,
        ppm,
        dbe,
        float(precursor_mz),
        neutral_mass,
        neutral_mass - math.floor(neutral_mass),
    ]

    values.extend(
        float(element_counts[e])
        for e in ELEMENTS
    )

    values.extend([
        float(total_atoms),
        float(heavy_atoms),
        float(hetero_atoms),
        float(halogen_atoms),
        float(rare_types),
        safe_ratio(h, c),
        safe_ratio(n, c),
        safe_ratio(o, c),
        safe_ratio(hetero_atoms, c),
        safe_ratio(dbe, c) if not math.isnan(dbe) else math.nan,

        float(v3_present),
        rank_numeric(v3_rank),
        rank_norm(v3_rank, v3_top),
        rr(v3_rank),

        float(old_present),
        rank_numeric(old_rank),
        rank_norm(old_rank, old_top),
        rr(old_rank),

        float(direct_present),
        rank_numeric(direct_rank),
        rank_norm(direct_rank, direct_top),
        rr(direct_rank),

        float(f_present),
        rank_numeric(f_rank),
        rank_norm(f_rank, f_top),
        rr(f_rank),

        float(support),
        float(best_rank_norm),
        float(rr_sum),
        float(is_chno_only),
        float(str(polarity).lower().startswith("pos")),

        float(spec["spec_num_peaks"]),
        float(spec["spec_base_mz_ratio"]),
        float(spec["spec_weighted_mz_ratio"]),
        float(spec["spec_high_mz_fraction"]),
        float(spec["spec_top5_intensity_fraction"]),
    ])

    values.extend(
        float(best_adduct == a)
        for a in adduct_names
    )

    return np.asarray(values, dtype=np.float32)


# =============================================================================
# Worker
# =============================================================================

BASE = None
BUILDER = None
RULE = None
ARGS = None
LIMITS = None
SIGNATURES = None
ADDUCT_NAMES = None
FEATURE_NAMES = None


class SampleTimeout(Exception):
    pass


def alarm_handler(signum, frame):
    raise SampleTimeout()


def init_worker(
    base_file: str,
    builder_file: str,
    rule_file: str,
    args_dict: Dict[str, Any],
    limits: Dict[str, int],
    signatures,
    adduct_names,
    feature_names,
):
    global BASE, BUILDER, RULE, ARGS, LIMITS, SIGNATURES, ADDUCT_NAMES, FEATURE_NAMES

    BASE = import_module_from_file(
        "preranker_builder_base_worker",
        base_file,
    )
    BUILDER = import_module_from_file(
        "preranker_builder_sft_worker",
        builder_file,
    )
    RULE = BASE.load_rule_module(rule_file)

    ARGS = argparse.Namespace(**args_dict)
    LIMITS = limits
    SIGNATURES = list(signatures)
    ADDUCT_NAMES = list(adduct_names)
    FEATURE_NAMES = list(feature_names)


def worker(payload):
    sample_order, fold_idx, item = payload
    started = time.perf_counter()

    if ARGS.sample_timeout > 0:
        signal.signal(signal.SIGALRM, alarm_handler)
        signal.alarm(ARGS.sample_timeout)

    try:
        identifier = BASE.get_identifier(item, fold_idx)
        true_formula = BASE.get_true_formula(item)

        # Evaluation fallback: preserve the raw annotated formula even when
        # it contains elements outside the candidate generator's supported
        # element space. This value is used only as the evaluation label.
        if not true_formula and isinstance(item, dict):
            molecule = item.get("molecule", {})
            if isinstance(molecule, dict):
                true_formula = str(molecule.get("formula") or "").strip()
        precursor_mz = BASE.get_precursor_mz(item)
        polarity = BASE.get_polarity(item)

        # Fallback for MassSpecGym records without explicit polarity.
        # Only use the charge sign of metadata.adduct, not adduct identity.
        if not polarity:
            metadata = item.get("metadata", {}) if isinstance(item, dict) else {}
            annotated_adduct = metadata.get("adduct", "")
            adduct_text = str(annotated_adduct).strip()

            if adduct_text.endswith("+"):
                polarity = "positive"
            elif adduct_text.endswith("-"):
                polarity = "negative"

        if precursor_mz is None or not polarity or not true_formula:
            return {
                "fold_idx": fold_idx,
                "sample_order": sample_order,
                "identifier": identifier,
                "status": "skip_missing",
                "kept": 0,
                "elapsed_sec": time.perf_counter() - started,
            }

        mzs, intensities = BASE.get_fragments_and_intensities(item)

        heuristic_frags, _ = BASE.prepare_fragments(
            precursor_mz,
            mzs,
            intensities,
            ARGS.heuristic_max_fragments,
        )

        v3_current, v3_broad_full = build_v3_current_and_broad(
            BASE,
            precursor_mz,
            polarity,
            heuristic_frags,
            LIMITS,
            ARGS,
            SIGNATURES,
        )

        old_full = BASE.build_old(
            precursor_mz,
            polarity,
            heuristic_frags,
            LIMITS,
            ARGS,
        )

        direct_full = BASE.build_direct(
            RULE,
            precursor_mz,
            polarity,
            mzs,
            intensities,
            ARGS,
        )

        f_current = [
            f for f in v3_current
            if contains_element(BASE, f, "F")
        ]
        f_broad_full = [
            f for f in v3_broad_full
            if contains_element(BASE, f, "F")
        ]

        v3 = v3_broad_full[:ARGS.v3_top]
        old = old_full[:ARGS.old_top]
        direct = direct_full[:ARGS.direct_top]
        f_broad = f_broad_full[:ARGS.f_top]

        practical = ordered_union(
            BASE,
            v3,
            old,
            direct,
            f_broad,
        )

        compact = ordered_union(
            BASE,
            v3_current[:ARGS.v3_short],
            old_full[:ARGS.old_short],
            direct_full[:ARGS.direct_short],
            f_current[:ARGS.f_short],
        )

        true_key = BASE.formula_key(true_formula)
        practical_keys = [
            BASE.formula_key(x)
            for x in practical
        ]
        gold_in_pool = int(
            true_key is not None
            and true_key in practical_keys
        )

        if ARGS.drop_no_gold and not gold_in_pool:
            return {
                "fold_idx": fold_idx,
                "sample_order": sample_order,
                "identifier": identifier,
                "status": "drop_no_gold",
                "kept": 0,
                "gold_in_pool": 0,
                "candidate_count": len(practical),
                "elapsed_sec": time.perf_counter() - started,
            }

        v3_ranks = rank_map(BASE, v3)
        old_ranks = rank_map(BASE, old)
        direct_ranks = rank_map(BASE, direct)
        f_ranks = rank_map(BASE, f_broad)

        spec = spectrum_summary(
            float(precursor_mz),
            mzs,
            intensities,
        )

        X_rows = []
        y_rows = []

        for formula in practical:
            key = BASE.formula_key(formula)

            feat = candidate_features(
                BASE,
                BUILDER,
                formula,
                float(precursor_mz),
                polarity,
                spec,
                v3_ranks.get(key),
                old_ranks.get(key),
                direct_ranks.get(key),
                f_ranks.get(key),
                ARGS.v3_top,
                ARGS.old_top,
                ARGS.direct_top,
                ARGS.f_top,
                ADDUCT_NAMES,
            )

            X_rows.append(feat)
            y_rows.append(
                int(
                    true_key is not None
                    and key == true_key
                )
            )

        X = np.vstack(X_rows).astype(
            np.float32,
            copy=False,
        )
        y = np.asarray(
            y_rows,
            dtype=np.uint8,
        )

        prompt_mz, prompt_int = BASE.prepare_fragments(
            precursor_mz,
            mzs,
            intensities,
            ARGS.prompt_max_peaks,
        )

        if prompt_int:
            mx = max(prompt_int) or 1.0
            prompt_peaks = [
                [
                    float(m),
                    float(i) / float(mx) * 100.0,
                ]
                for m, i in zip(
                    prompt_mz,
                    prompt_int,
                )
            ]
        else:
            prompt_peaks = []

        meta = {
            "fold_idx": fold_idx,
            "sample_order": sample_order,
            "identifier": identifier,
            "precursor_mz": float(precursor_mz),
            "polarity": polarity,
            "prompt_peaks": prompt_peaks,

            # Evaluation-only fields.
            "true_formula": true_formula,
            "true_adduct": get_true_adduct(item),

            "gold_in_pool": gold_in_pool,
            "candidate_count": len(practical),
            "formulas": practical,
            "compact_formulas": compact,

            # Source lists retained so later shortlist export can rebuild
            # the exact Qwen candidate metadata without regenerating chemistry.
            "v3_practical": v3,
            "old_practical": old,
            "direct_practical": direct,
            "f_practical": f_broad,

            "v3_current_top10": v3_current[:ARGS.v3_short],
            "old_current_top10": old_full[:ARGS.old_short],
            "direct_current_top5": direct_full[:ARGS.direct_short],
            "f_current_top15": f_current[:ARGS.f_short],
        }

        return {
            "fold_idx": fold_idx,
            "sample_order": sample_order,
            "identifier": identifier,
            "status": "ok",
            "kept": 1,
            "gold_in_pool": gold_in_pool,
            "candidate_count": len(practical),
            "X": X,
            "y": y,
            "meta": meta,
            "elapsed_sec": time.perf_counter() - started,
        }

    except SampleTimeout:
        return {
            "fold_idx": fold_idx,
            "sample_order": sample_order,
            "identifier": BASE.get_identifier(item, fold_idx),
            "status": "timeout",
            "kept": 0,
            "elapsed_sec": time.perf_counter() - started,
        }

    except Exception as exc:
        return {
            "fold_idx": fold_idx,
            "sample_order": sample_order,
            "identifier": BASE.get_identifier(item, fold_idx),
            "status": f"error:{repr(exc)}",
            "kept": 0,
            "elapsed_sec": time.perf_counter() - started,
        }

    finally:
        if ARGS.sample_timeout > 0:
            signal.alarm(0)


# =============================================================================
# Main
# =============================================================================

def main():
    p = argparse.ArgumentParser()

    p.add_argument("--base-file", required=True)
    p.add_argument("--builder-file", required=True)
    p.add_argument("--rule-file", required=True)
    p.add_argument("--input", required=True)

    p.add_argument("--fold", required=True)
    p.add_argument("--sample-size", type=int, default=5000)
    p.add_argument("--sample-seed", type=int, default=20260818)

    p.add_argument("--v3-top", type=int, default=50)
    p.add_argument("--old-top", type=int, default=400)
    p.add_argument("--direct-top", type=int, default=5)
    p.add_argument("--f-top", type=int, default=15)

    p.add_argument("--v3-ppm", type=float, default=10.0)
    p.add_argument("--old-ppm", type=float, default=5.0)
    p.add_argument("--direct-ppm", type=float, default=10.0)

    p.add_argument("--v3-source-top", type=int, default=200)
    p.add_argument("--old-source-top", type=int, default=200)

    p.add_argument("--v3-short", type=int, default=10)
    p.add_argument("--old-short", type=int, default=10)
    p.add_argument("--direct-short", type=int, default=5)
    p.add_argument("--f-short", type=int, default=15)

    p.add_argument("--heuristic-max-fragments", type=int, default=50)
    p.add_argument("--direct-max-fragments", type=int, default=0)
    p.add_argument("--prompt-max-peaks", type=int, default=30)

    p.add_argument(
        "--max-counts",
        default="C=80,H=160,N=20,O=30,P=6,S=8,F=30,Cl=6,Br=4,I=3,Si=4",
    )
    p.add_argument("--extension-min-train-count", type=int, default=1)
    p.add_argument("--max-extension-signatures", type=int, default=32)

    p.add_argument(
        "--drop-no-gold",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--sample-timeout", type=int, default=300)
    p.add_argument("--progress-every", type=int, default=50)

    p.add_argument(
        "--output-prefix",
        required=True,
        help="e.g. ./outputs/preranker_train5000",
    )
    p.add_argument("--resume", action="store_true")

    args = p.parse_args()

    base = import_module_from_file(
        "preranker_builder_base_parent",
        args.base_file,
    )
    builder = import_module_from_file(
        "preranker_builder_sft_parent",
        args.builder_file,
    )
    _ = base.load_rule_module(
        args.rule_file
    )

    all_records = base.load_records(
        args.input
    )

    signatures, _ = base.build_extension_signatures(
        all_records,
        args.extension_min_train_count,
        args.max_extension_signatures,
    )

    limits = base.parse_limits(
        args.max_counts
    )

    fold_records = [
        x for x in all_records
        if str(base.get_fold(x)).lower()
        == args.fold.lower()
    ]

    if not fold_records:
        raise RuntimeError(
            f"No records for fold={args.fold!r}"
        )

    if (
        args.sample_size <= 0
        or args.sample_size >= len(fold_records)
    ):
        indices = list(
            range(len(fold_records))
        )
    else:
        rng = random.Random(
            args.sample_seed
        )
        indices = sorted(
            rng.sample(
                range(len(fold_records)),
                args.sample_size,
            )
        )

    payloads = [
        (
            sample_order,
            idx + 1,
            fold_records[idx],
        )
        for sample_order, idx in enumerate(
            indices,
            1,
        )
    ]

    # Stable adduct vocabulary.
    adduct_names = sorted(
        {
            canonical_adduct(adduct.name)
            for polarity in base.ADDUCTS
            for adduct in base.ADDUCTS[polarity]
        }
    )

    feature_names = build_feature_names(
        adduct_names
    )

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    x_path = Path(str(prefix) + ".X.f32.bin")
    y_path = Path(str(prefix) + ".y.u8.bin")
    group_path = Path(str(prefix) + ".groups.i32.bin")
    meta_path = Path(str(prefix) + ".meta.jsonl.gz")
    audit_path = Path(str(prefix) + ".audit.jsonl")
    feature_path = Path(str(prefix) + ".features.json")
    dataset_path = Path(str(prefix) + ".dataset.json")

    completed = (
        load_completed_audit(audit_path)
        if args.resume
        else set()
    )

    remaining = [
        x for x in payloads
        if int(x[1]) not in completed
    ]

    if not args.resume:
        for path in (
            x_path,
            y_path,
            group_path,
            audit_path,
            feature_path,
            dataset_path,
        ):
            if path.exists():
                path.unlink()

        if meta_path.exists():
            meta_path.unlink()

        feature_path.write_text(
            json.dumps(
                {
                    "feature_names": feature_names,
                    "adduct_names": adduct_names,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    print("=" * 112)
    print("BUILD BROAD PRE-RANKER DATASET")
    print("=" * 112)
    print(f"Fold                    : {args.fold}")
    print(f"Fold size               : {len(fold_records)}")
    print(f"Selected spectra        : {len(payloads)}")
    print(f"Remaining               : {len(remaining)}")
    print(f"Random seed             : {args.sample_seed}")
    print(f"Practical pool          : V3Top{args.v3_top} + OLDTop{args.old_top} + DirectTop{args.direct_top} + FTop{args.f_top}")
    print(f"Feature dim             : {len(feature_names)}")
    print(f"Drop no-gold groups     : {args.drop_no_gold}")
    print(f"Workers                 : {args.num_workers}")
    print("LLM                     : NO")
    print("SIRIUS                  : NO")
    print("External DB             : NO")
    print("Gold-dependent routing  : NO")
    print("=" * 112, flush=True)

    mode = "ab" if args.resume else "wb"
    audit_mode = "a" if args.resume else "w"
    meta_mode = "at" if args.resume else "wt"

    started = time.perf_counter()
    status_counter = Counter()
    candidate_counts = []
    kept_groups_this_run = 0
    kept_rows_this_run = 0
    gold_hits_all_success = 0
    successful_built = 0

    ctx = mp.get_context("fork")

    with x_path.open(mode) as fx, \
         y_path.open(mode) as fy, \
         group_path.open(mode) as fg, \
         audit_path.open(audit_mode, encoding="utf-8") as faudit, \
         gzip.open(meta_path, meta_mode, encoding="utf-8") as fmeta, \
         ctx.Pool(
             processes=min(
                 args.num_workers,
                 max(1, len(remaining)),
             ),
             initializer=init_worker,
             initargs=(
                 args.base_file,
                 args.builder_file,
                 args.rule_file,
                 vars(args).copy(),
                 limits,
                 signatures,
                 adduct_names,
                 feature_names,
             ),
         ) as pool:

        for done, result in enumerate(
            pool.imap_unordered(
                worker,
                remaining,
                chunksize=1,
            ),
            1,
        ):
            status = str(result.get("status", ""))
            status_counter[status] += 1

            if status == "ok":
                successful_built += 1
                gold_hits_all_success += int(
                    result.get("gold_in_pool", 0)
                )

            if result.get("candidate_count") is not None:
                candidate_counts.append(
                    int(result["candidate_count"])
                )

            if int(result.get("kept", 0)) == 1:
                X = result["X"]
                y = result["y"]

                if X.shape[1] != len(feature_names):
                    raise RuntimeError(
                        f"Feature dim mismatch: {X.shape}"
                    )

                X.astype(
                    np.float32,
                    copy=False,
                ).tofile(fx)

                y.astype(
                    np.uint8,
                    copy=False,
                ).tofile(fy)

                np.asarray(
                    [len(y)],
                    dtype=np.int32,
                ).tofile(fg)

                fmeta.write(
                    json.dumps(
                        result["meta"],
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                kept_groups_this_run += 1
                kept_rows_this_run += len(y)

            audit_row = {
                "fold_idx": int(result["fold_idx"]),
                "sample_order": int(result["sample_order"]),
                "identifier": result.get("identifier", ""),
                "status": status,
                "kept": int(result.get("kept", 0)),
                "gold_in_pool": int(result.get("gold_in_pool", 0)),
                "candidate_count": result.get("candidate_count"),
                "elapsed_sec": result.get("elapsed_sec"),
            }

            faudit.write(
                json.dumps(
                    audit_row,
                    ensure_ascii=False,
                )
                + "\n"
            )

            # Flush frequently so resume is safe.
            fx.flush()
            fy.flush()
            fg.flush()
            fmeta.flush()
            faudit.flush()

            if (
                args.progress_every > 0
                and done % args.progress_every == 0
            ):
                elapsed = time.perf_counter() - started
                print(
                    f"Processed {len(completed)+done}/{len(payloads)} "
                    f"| kept_this_run={kept_groups_this_run} "
                    f"| rows={kept_rows_this_run} "
                    f"| wall={elapsed/3600:.2f} h",
                    flush=True,
                )

    # Final counts from files, including previous resume runs.
    group_sizes = np.fromfile(
        group_path,
        dtype=np.int32,
    )

    n_groups = int(len(group_sizes))
    n_rows = int(group_sizes.sum())

    x_bytes_expected = (
        n_rows
        * len(feature_names)
        * np.dtype(np.float32).itemsize
    )

    y_bytes_expected = (
        n_rows
        * np.dtype(np.uint8).itemsize
    )

    if x_path.stat().st_size != x_bytes_expected:
        raise RuntimeError(
            f"X binary size mismatch: "
            f"{x_path.stat().st_size} vs {x_bytes_expected}"
        )

    if y_path.stat().st_size != y_bytes_expected:
        raise RuntimeError(
            f"y binary size mismatch: "
            f"{y_path.stat().st_size} vs {y_bytes_expected}"
        )

    meta_groups = count_gzip_lines(
        meta_path
    )

    if meta_groups != n_groups:
        raise RuntimeError(
            f"Metadata group mismatch: "
            f"{meta_groups} vs {n_groups}"
        )

    # Re-read audit for final summary.
    final_status = Counter()
    final_selected = 0
    final_kept = 0
    final_success = 0
    final_gold_hits = 0
    final_candidate_counts = []

    with audit_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            final_selected += 1
            final_status[
                str(row.get("status", ""))
            ] += 1
            final_kept += int(
                row.get("kept", 0)
            )

            if row.get("status") == "ok":
                final_success += 1
                final_gold_hits += int(
                    row.get("gold_in_pool", 0)
                )

            if row.get("candidate_count") is not None:
                final_candidate_counts.append(
                    int(row["candidate_count"])
                )

    dataset_info = {
        "fold": args.fold,
        "fold_size": len(fold_records),
        "selected_spectra": len(payloads),
        "sample_seed": args.sample_seed,

        "n_groups": n_groups,
        "n_rows": n_rows,
        "feature_dim": len(feature_names),

        "drop_no_gold": args.drop_no_gold,

        "config": {
            "v3_top": args.v3_top,
            "old_top": args.old_top,
            "direct_top": args.direct_top,
            "f_top": args.f_top,
            "v3_ppm": args.v3_ppm,
            "old_ppm": args.old_ppm,
            "direct_ppm": args.direct_ppm,
        },

        "statuses": dict(final_status),

        "successful_candidate_generation": final_success,
        "gold_hits_among_successful": final_gold_hits,
        "practical_pool_coverage_among_successful": (
            final_gold_hits / final_success
            if final_success else 0.0
        ),

        "candidate_count": {
            "mean": (
                statistics.mean(final_candidate_counts)
                if final_candidate_counts else None
            ),
            "median": (
                statistics.median(final_candidate_counts)
                if final_candidate_counts else None
            ),
            "p95": percentile(
                final_candidate_counts,
                0.95,
            ),
            "max": (
                max(final_candidate_counts)
                if final_candidate_counts else None
            ),
        },

        "paths": {
            "X": str(x_path.resolve()),
            "y": str(y_path.resolve()),
            "groups": str(group_path.resolve()),
            "meta": str(meta_path.resolve()),
            "audit": str(audit_path.resolve()),
            "features": str(feature_path.resolve()),
        },
    }

    dataset_path.write_text(
        json.dumps(
            dataset_info,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 112)
    print("DATASET SUMMARY")
    print("=" * 112)
    print(f"Selected spectra          : {len(payloads)}")
    print(f"Kept ranking groups       : {n_groups}")
    print(f"Candidate rows            : {n_rows}")
    print(f"Feature dim               : {len(feature_names)}")
    print(f"Statuses                  : {dict(final_status)}")
    print(
        f"Practical coverage        : "
        f"{final_gold_hits}/{final_success} "
        f"= {100*final_gold_hits/final_success if final_success else 0:.2f}% "
        f"(among successful builds)"
    )
    if final_candidate_counts:
        print(
            f"Candidate count           : "
            f"mean={statistics.mean(final_candidate_counts):.1f}, "
            f"median={statistics.median(final_candidate_counts):.1f}, "
            f"P95={percentile(final_candidate_counts,0.95):.1f}, "
            f"max={max(final_candidate_counts)}"
        )

    print("\nSaved:")
    for path in (
        x_path,
        y_path,
        group_path,
        meta_path,
        audit_path,
        feature_path,
        dataset_path,
    ):
        print(" ", path.resolve())


if __name__ == "__main__":
    main()
