#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build resumable no-thinking listwise SFT data for molecular-formula candidate reranking.

LOCKED candidate pool
=====================
For every spectrum:

    V3 global Top-10
    ∪ OLD Top-10
    ∪ Direct Top-5
    ∪ V3-F reserved Top-15
    -> deduplicate by neutral molecular formula

The candidate pool is the locked version used for the final candidate-pool
ablation. This script DOES NOT use SIRIUS rescue and DOES NOT use gold labels
to route candidate generation.

Training objective
==================
No chain-of-thought is generated.

Input:
    precursor m/z
    polarity
    top MS/MS peaks
    shuffled final candidate formulas
    candidate-source/rank metadata available at inference time

Target:
    {"ranked_candidates":[
        {"rank":1,"candidate_id":"Cxx","formula":"...","adduct":"..."},
        ...
    ]}

The target is LISTWISE but contains NO chain-of-thought.
The supervised gold formula is always placed at rank 1. The remaining output
positions are filled by a deterministic weak ranking that uses only
inference-available candidate evidence (branch support/ranks and precursor ppm),
not reference-relative features. This makes the target compatible with direct
Top-1/Top-3/Top-5 evaluation while avoiding teacher-generated reasoning.

Candidate IDs are assigned only AFTER deterministic input shuffling, so the
model cannot exploit a fixed candidate position or branch ordering shortcut.

Candidate-order leakage prevention
=====================================
Candidate rows are deterministically shuffled per spectrum. Candidate IDs are
assigned only after this shuffle. The model therefore cannot learn that a fixed
branch position or the first candidate is usually preferred. The branch/rank
signals are provided explicitly instead:
    v3=<rank or ->
    old=<rank or ->
    direct=<rank or ->
    f=<rank or ->
    support=<number of supporting lanes>

Sampling strategy
=================
Default balanced sampling for the eventual larger training set:
    50% standard
    30% hard_nonrare
    20% rare

rare:
    gold contains F/I/Si

hard_nonrare:
    gold does NOT contain F/I/Si,
    gold is absent from V3 global Top-10,
    but gold is rescued by OLD / Direct / another final-pool lane

standard:
    all other gold-covered training examples

For --target-examples 10, the default quotas become:
    5 standard + 3 hard_nonrare + 2 rare

Recommended later run:
    --target-examples 30000

Leakage policy
==============
- Train gold formula is used only as the SFT target and for offline sampling
  category assignment. That is supervised training, not inference routing.
- Candidate generation itself is identical for every sample.
- Exact annotated adduct is never supplied to candidate generation.
- If explicit polarity is missing, only adduct SIGN may be used.
- F lane is activated for EVERY sample.
- No teacher model.
- No chain-of-thought.
- No SIRIUS rescue.

Implementation
==============
Imports the locked OLD/V3/Direct chemistry from:
    eval_final_candidate_union_2000.py
and imports Direct-rule logic from:
    mz_formula_candidates.py

Python 3.13 compatible.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
import random
import signal
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


# =============================================================================
# Dynamic imports
# =============================================================================

def import_module_from_file(module_name: str, file_path: str):
    path = Path(file_path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    spec = importlib.util.spec_from_file_location(module_name, str(path))

    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import: {path}")

    module = importlib.util.module_from_spec(spec)

    # Required for Python 3.13 dataclass imports.
    sys.modules[spec.name] = module

    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise

    return module


# =============================================================================
# Formula/candidate helpers
# =============================================================================

def contains_element(base, formula: str, element: str) -> bool:
    counts = base.parse_formula(formula)
    return bool(counts) and counts.get(element, 0) > 0


def ordered_union(base, *lists: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()

    for values in lists:
        for formula in values:
            key = base.formula_key(formula)

            if key is None or key in seen:
                continue

            seen.add(key)
            out.append(formula)

    return out


def find_rank(base, formula: str, formulas: Sequence[str]) -> Optional[int]:
    target = base.formula_key(formula)

    for rank, value in enumerate(formulas, 1):
        if base.formula_key(value) == target:
            return rank

    return None


def contains_true(base, formulas: Sequence[str], true_formula: str) -> bool:
    return find_rank(base, true_formula, formulas) is not None


def stable_seed(identifier: str, global_seed: int) -> int:
    payload = f"{global_seed}|{identifier}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


# =============================================================================
# Prompt construction
# =============================================================================

SYSTEM_PROMPT = """你是一名 LC-MS/MS 中性分子式候选重排序专家。

你的任务是根据 precursor m/z、离子模式、MS/MS 碎片峰以及候选分子式的确定性化学证据，对候选表中的中性分子式进行重新排序。

重要约束：
1. 只能使用候选表中已有的 candidate_id，不允许生成候选表之外的分子式。
2. 真实 adduct 不会作为输入提供。候选中的 BestAdduct 仅表示基于 precursor 质量得到的最匹配潜在加合物解释。
3. V3Rank、OldRank、DirectRank 和 FRank 是不同规则路径提供的先验排序，只能作为辅助证据，不能机械照抄。
4. Support 表示独立候选生成器的支持数量，其中 V3 global 与 V3-F reserved lane 属于同一个 V3 生成器，不重复计票。
5. 不要输出推理过程、分析过程、<think> 或解释。
6. 如果候选数不少于 10，必须输出恰好 10 个不同候选；如果候选总数少于 10，则输出全部候选。
7. ranked_candidates 必须按从最可能到最不可能排序，rank 必须从 1 连续编号。
8. 每个 candidate_id、formula、adduct 必须与输入候选表中的对应字段严格一致。

最终只输出一个 JSON 对象，结构如下：
{"ranked_candidates":[{"rank":1,"candidate_id":"C01","formula":"C10H12N2O3","adduct":"[M+H]+"}]}"""


def select_prompt_peaks(
    base,
    precursor_mz: float,
    mzs: Sequence[float],
    intensities: Sequence[float],
    max_peaks: int,
) -> List[Tuple[float, float]]:
    """
    Keep the most intense peaks, then sort them by m/z.
    Intensities are normalized to max=100.
    """
    mz_sel, int_sel = base.prepare_fragments(
        precursor_mz,
        mzs,
        intensities,
        max_peaks,
    )

    if not mz_sel:
        return []

    max_intensity = max(int_sel) if int_sel else 1.0

    if max_intensity <= 0:
        max_intensity = 1.0

    return [
        (float(mz), float(intensity) / max_intensity * 100.0)
        for mz, intensity in zip(mz_sel, int_sel)
    ]


def best_mass_matched_adduct(
    base,
    formula: str,
    precursor_mz: float,
    polarity: str,
) -> Dict[str, Any]:
    """
    Deterministic inference-time chemical evidence.

    For a neutral molecular formula, evaluate all latent adduct hypotheses
    allowed for the observed polarity and select the one with the smallest
    absolute precursor ppm error.

    This does NOT use the annotated/true adduct.
    """
    counts = base.parse_formula(formula)

    if not counts:
        return {
            "best_adduct": "-",
            "ppm_error": None,
            "dbe": None,
        }

    neutral_mass = base.exact_mass(counts)
    dbe = base.calc_dbe(counts)

    best = None

    for adduct in base.ADDUCTS[polarity]:
        calculated_mz = neutral_mass + adduct.mass_shift
        ppm_error = (
            (calculated_mz - precursor_mz)
            / precursor_mz
            * 1_000_000.0
        )

        record = {
            "best_adduct": adduct.name,
            "ppm_error": ppm_error,
            "dbe": dbe,
        }

        if (
            best is None
            or abs(ppm_error) < abs(best["ppm_error"])
        ):
            best = record

    return best or {
        "best_adduct": "-",
        "ppm_error": None,
        "dbe": dbe,
    }


def build_candidate_metadata(
    base,
    precursor_mz: float,
    polarity: str,
    final_candidates: Sequence[str],
    v3_short: Sequence[str],
    old_short: Sequence[str],
    direct_short: Sequence[str],
    f_short: Sequence[str],
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    for formula in final_candidates:
        v3_rank = find_rank(base, formula, v3_short)
        old_rank = find_rank(base, formula, old_short)
        direct_rank = find_rank(base, formula, direct_short)
        f_rank = find_rank(base, formula, f_short)

        # F lane is derived from the V3 full pool, so V3 and F must NOT be
        # counted as two independent votes.
        v3_supported = (
            v3_rank is not None
            or f_rank is not None
        )

        support = (
            int(v3_supported)
            + int(old_rank is not None)
            + int(direct_rank is not None)
        )

        chemistry = best_mass_matched_adduct(
            base,
            formula,
            precursor_mz,
            polarity,
        )

        records.append({
            "formula": formula,
            "best_adduct": chemistry["best_adduct"],
            "ppm_error": chemistry["ppm_error"],
            "dbe": chemistry["dbe"],
            "v3_rank": v3_rank,
            "old_rank": old_rank,
            "direct_rank": direct_rank,
            "f_rank": f_rank,
            "support": support,
        })

    return records



def weak_negative_key(candidate: Dict[str, Any]) -> Tuple[Any, ...]:
    """
    Reference-independent weak ordering for NEGATIVES only.

    The gold item is supervised separately at rank 1.
    Remaining candidates are ordered using only evidence available at inference:
      1) more independent-generator support first;
      2) better best rank across V3/OLD/Direct/F lane;
      3) better sum of reciprocal branch ranks;
      4) smaller absolute precursor ppm error;
      5) deterministic formula tie-break.

    This avoids teacher-model reasoning and avoids using the reference formula
    to define rank 2..K.
    """
    ranks = [
        candidate.get("v3_rank"),
        candidate.get("old_rank"),
        candidate.get("direct_rank"),
        candidate.get("f_rank"),
    ]

    valid_ranks = [r for r in ranks if r is not None]
    best_rank = min(valid_ranks) if valid_ranks else 10**9

    reciprocal_rank_sum = sum(
        1.0 / float(r)
        for r in valid_ranks
        if r and r > 0
    )

    ppm_error = candidate.get("ppm_error")
    abs_ppm = (
        abs(float(ppm_error))
        if ppm_error is not None
        else float("inf")
    )

    return (
        -int(candidate.get("support", 0)),
        best_rank,
        -reciprocal_rank_sum,
        abs_ppm,
        str(candidate.get("formula", "")),
    )


def build_listwise_target(
    base,
    candidate_records: Sequence[Dict[str, Any]],
    true_formula: str,
    output_rank_k: int,
) -> List[Dict[str, Any]]:
    """
    Create no-thinking listwise supervision.

    rank 1:
        supervised gold formula.

    ranks 2..K:
        highest-priority negatives under weak_negative_key(), which is fully
        reference-independent and uses only inference-time candidate evidence.
    """
    true_key = base.formula_key(true_formula)

    gold = None
    negatives: List[Dict[str, Any]] = []

    for candidate in candidate_records:
        if base.formula_key(candidate["formula"]) == true_key:
            gold = candidate
        else:
            negatives.append(candidate)

    if gold is None:
        raise RuntimeError(
            "Gold formula is not present in candidate_records."
        )

    negatives = sorted(
        negatives,
        key=weak_negative_key,
    )

    target_count = min(
        int(output_rank_k),
        len(candidate_records),
    )

    selected = [gold] + negatives[: max(0, target_count - 1)]

    ranked: List[Dict[str, Any]] = []

    for rank, candidate in enumerate(selected, 1):
        ranked.append({
            "rank": rank,
            "candidate_id": candidate["candidate_id"],
            "formula": candidate["formula"],
            "adduct": candidate["best_adduct"],
        })

    return ranked

def build_user_prompt(
    precursor_mz: float,
    polarity: str,
    peaks: Sequence[Tuple[float, float]],
    candidates: Sequence[Dict[str, Any]],
) -> str:
    lines: List[str] = []

    lines.append("【谱图信息】")
    lines.append(f"precursor_mz: {precursor_mz:.6f}")
    lines.append(f"polarity: {polarity}")
    lines.append("")

    lines.append(
        "【MS/MS 碎片峰】"
    )
    lines.append(
        "以下峰按强度选取后按 m/z 排序，强度已归一化到 0-100："
    )

    if peaks:
        peak_text = ", ".join(
            f"{mz:.5f}:{intensity:.1f}"
            for mz, intensity in peaks
        )
        lines.append(peak_text)
    else:
        lines.append("none")

    lines.append("")
    lines.append("【候选中性分子式】")
    lines.append(
        "字段说明：BestAdduct 为按 precursor 质量误差选择的最匹配潜在加合物；"
        "ppm 为对应 precursor 质量误差；DBE 为候选分子式的不饱和度；"
        "V3Rank/OldRank/DirectRank/FRank 为不同候选路径中的排名；"
        "Support 为独立生成器支持数，V3 与 F lane 不重复计票。"
    )
    lines.append("")

    def rank_text(value):
        return "-" if value is None else str(value)

    def float_text(value, digits=3):
        return "-" if value is None else f"{value:.{digits}f}"

    for candidate in candidates:
        lines.append(
            f"{candidate['candidate_id']} | "
            f"Formula={candidate['formula']} | "
            f"BestAdduct={candidate['best_adduct']} | "
            f"ppm={float_text(candidate['ppm_error'], 3)} | "
            f"DBE={float_text(candidate['dbe'], 1)} | "
            f"V3Rank={rank_text(candidate['v3_rank'])} | "
            f"OldRank={rank_text(candidate['old_rank'])} | "
            f"DirectRank={rank_text(candidate['direct_rank'])} | "
            f"FRank={rank_text(candidate['f_rank'])} | "
            f"Support={candidate['support']}"
        )

    lines.append("")
    lines.append(
        "请综合 precursor 质量一致性、化学合理性、不同候选路径的支持情况以及 "
        "MS/MS 碎片信息，对候选进行重新排序。"
    )
    lines.append(
        "如果候选数不少于 10，请输出恰好 10 个不同候选；"
        "如果候选总数少于 10，则输出全部候选。"
    )
    lines.append(
        "不要输出推理、解释或 <think>。只输出 JSON。JSON 字段为 ranked_candidates，"
        "其中每个元素必须包含 rank、candidate_id、formula、adduct。"
    )

    return "\n".join(lines)


# =============================================================================
# Sampling categories
# =============================================================================

def classify_example(
    base,
    true_formula: str,
    v3_short: Sequence[str],
    final_candidates: Sequence[str],
) -> str:
    counts = base.parse_formula(true_formula) or {}

    if any(counts.get(e, 0) > 0 for e in ("F", "I", "Si")):
        return "rare"

    # Candidate pool must already contain gold before this function is called.
    if not contains_true(base, v3_short, true_formula):
        return "hard_nonrare"

    return "standard"


def compute_quotas(
    target_examples: int,
    rare_frac: float,
    hard_frac: float,
) -> Dict[str, int]:
    rare = int(round(target_examples * rare_frac))
    hard = int(round(target_examples * hard_frac))

    if rare + hard > target_examples:
        raise ValueError("rare_frac + hard_frac must be <= 1.")

    standard = target_examples - rare - hard

    return {
        "standard": standard,
        "hard_nonrare": hard,
        "rare": rare,
    }


# =============================================================================
# Worker state
# =============================================================================

BASE = None
RULE = None
ARGS = None
LIMITS = None
SIGNATURES = None


class SampleTimeout(Exception):
    pass


def alarm_handler(signum, frame):
    raise SampleTimeout()


def init_worker(
    base_file: str,
    rule_file: str,
    args_dict: Dict[str, Any],
    limits: Dict[str, int],
    signatures,
):
    global BASE, RULE, ARGS, LIMITS, SIGNATURES

    BASE = import_module_from_file(
        "formula_sft_base_worker",
        base_file,
    )

    RULE = BASE.load_rule_module(rule_file)
    ARGS = argparse.Namespace(**args_dict)
    LIMITS = limits
    SIGNATURES = list(signatures)


def worker(payload):
    dataset_idx, item = payload

    if ARGS.sample_timeout > 0:
        signal.signal(signal.SIGALRM, alarm_handler)
        signal.alarm(ARGS.sample_timeout)

    started = time.perf_counter()

    try:
        true_formula = BASE.get_true_formula(item)
        precursor_mz = BASE.get_precursor_mz(item)
        polarity = BASE.get_polarity(item)

        if not true_formula or precursor_mz is None or not polarity:
            return {
                "status": "skip",
                "dataset_idx": dataset_idx,
                "reason": "missing_required_field",
            }

        mzs, intensities = BASE.get_fragments_and_intensities(item)

        heuristic_fragments, _ = BASE.prepare_fragments(
            precursor_mz,
            mzs,
            intensities,
            ARGS.heuristic_max_fragments,
        )

        # ---------------------------------------------------------------------
        # Locked candidate pool
        # ---------------------------------------------------------------------

        old_full = BASE.build_old(
            precursor_mz,
            polarity,
            heuristic_fragments,
            LIMITS,
            ARGS,
        )

        v3_full = BASE.build_v3(
            precursor_mz,
            polarity,
            heuristic_fragments,
            LIMITS,
            ARGS,
            SIGNATURES,
        )

        direct_full = BASE.build_direct(
            RULE,
            precursor_mz,
            polarity,
            mzs,
            intensities,
            ARGS,
        )

        old_short = old_full[:ARGS.old_short]
        v3_short = v3_full[:ARGS.v3_short]
        direct_short = direct_full[:ARGS.direct_short]

        f_lane_full = [
            formula
            for formula in v3_full
            if contains_element(BASE, formula, "F")
        ]
        f_short = f_lane_full[:ARGS.f_short]

        final_candidates = ordered_union(
            BASE,
            v3_short,
            old_short,
            direct_short,
            f_short,
        )

        if not contains_true(
            BASE,
            final_candidates,
            true_formula,
        ):
            return {
                "status": "gold_miss",
                "dataset_idx": dataset_idx,
                "identifier": BASE.get_identifier(item, dataset_idx),
                "true_formula": true_formula,
                "candidate_count": len(final_candidates),
                "elapsed_sec": time.perf_counter() - started,
            }

        category = classify_example(
            BASE,
            true_formula,
            v3_short,
            final_candidates,
        )

        # ---------------------------------------------------------------------
        # Candidate metadata + deterministic shuffle
        # ---------------------------------------------------------------------

        candidate_records = build_candidate_metadata(
            BASE,
            precursor_mz,
            polarity,
            final_candidates,
            v3_short,
            old_short,
            direct_short,
            f_short,
        )

        identifier = BASE.get_identifier(item, dataset_idx)

        if ARGS.shuffle_candidates:
            rng = random.Random(
                stable_seed(identifier, ARGS.seed)
            )
            rng.shuffle(candidate_records)

        # Candidate IDs are assigned AFTER shuffling.
        for index, candidate in enumerate(candidate_records, 1):
            candidate["candidate_id"] = f"C{index:02d}"

        gold_position = None
        gold_candidate_id = None

        for index, candidate in enumerate(candidate_records, 1):
            if (
                BASE.formula_key(candidate["formula"])
                == BASE.formula_key(true_formula)
            ):
                gold_position = index
                gold_candidate_id = candidate["candidate_id"]
                break

        if gold_candidate_id is None:
            raise RuntimeError(
                "Gold formula was present before shuffle but candidate_id "
                "could not be resolved after shuffle."
            )

        # ---------------------------------------------------------------------
        # Prompt peaks
        # ---------------------------------------------------------------------

        prompt_peaks = select_prompt_peaks(
            BASE,
            precursor_mz,
            mzs,
            intensities,
            ARGS.prompt_max_peaks,
        )

        user_prompt = build_user_prompt(
            precursor_mz,
            polarity,
            prompt_peaks,
            candidate_records,
        )

        ranked_target = build_listwise_target(
            BASE,
            candidate_records,
            true_formula,
            ARGS.output_rank_k,
        )

        assistant_target = json.dumps(
            {"ranked_candidates": ranked_target},
            ensure_ascii=False,
            separators=(",", ":"),
        )

        train_record = {
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
                {
                    "role": "assistant",
                    "content": assistant_target,
                },
            ]
        }

        audit_record = {
            "dataset_idx": dataset_idx,
            "identifier": identifier,
            "category": category,
            "true_formula": true_formula,
            "gold_position_after_shuffle": gold_position,
            "gold_candidate_id": gold_candidate_id,
            "precursor_mz": precursor_mz,
            "polarity": polarity,
            "candidate_count": len(candidate_records),
            "prompt_peak_count": len(prompt_peaks),

            "gold_in_v3_top10": int(
                contains_true(BASE, v3_short, true_formula)
            ),
            "gold_in_old_top10": int(
                contains_true(BASE, old_short, true_formula)
            ),
            "gold_in_direct_top5": int(
                contains_true(BASE, direct_short, true_formula)
            ),
            "gold_in_f_top15": int(
                contains_true(BASE, f_short, true_formula)
            ),

            "v3_short": v3_short,
            "old_short": old_short,
            "direct_short": direct_short,
            "f_short": f_short,
            "candidate_records_after_shuffle": candidate_records,
            "target_ranked_candidates": ranked_target,
            "prompt_peaks": prompt_peaks,
            "elapsed_sec": time.perf_counter() - started,
        }

        return {
            "status": "ok",
            "dataset_idx": dataset_idx,
            "identifier": identifier,
            "category": category,
            "train_record": train_record,
            "audit_record": audit_record,
        }

    except SampleTimeout:
        return {
            "status": "timeout",
            "dataset_idx": dataset_idx,
        }

    except Exception as exc:
        return {
            "status": "error",
            "dataset_idx": dataset_idx,
            "error": repr(exc),
        }

    finally:
        if ARGS.sample_timeout > 0:
            signal.alarm(0)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--base-file", required=True)
    parser.add_argument("--rule-file", required=True)
    parser.add_argument("--input", required=True)

    parser.add_argument("--fold", default="train")
    parser.add_argument("--target-examples", type=int, default=10)

    parser.add_argument(
        "--rare-frac",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--hard-frac",
        type=float,
        default=0.30,
    )

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--shuffle-candidates",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    # Locked candidate pool.
    parser.add_argument("--old-ppm", type=float, default=5.0)
    parser.add_argument("--v3-ppm", type=float, default=10.0)
    parser.add_argument("--direct-ppm", type=float, default=10.0)

    parser.add_argument("--old-source-top", type=int, default=200)
    parser.add_argument("--v3-source-top", type=int, default=200)

    parser.add_argument("--old-short", type=int, default=10)
    parser.add_argument("--v3-short", type=int, default=10)
    parser.add_argument("--direct-short", type=int, default=5)
    parser.add_argument("--f-short", type=int, default=15)

    parser.add_argument(
        "--max-counts",
        default="C=80,H=160,N=20,O=30,P=6,S=8,F=30,Cl=6,Br=4,I=3,Si=4",
    )

    parser.add_argument("--extension-min-train-count", type=int, default=1)
    parser.add_argument("--max-extension-signatures", type=int, default=32)

    parser.add_argument("--heuristic-max-fragments", type=int, default=50)
    parser.add_argument(
        "--direct-max-fragments",
        type=int,
        default=0,
        help="0 = all MS2 peaks for Direct candidate generation.",
    )

    parser.add_argument(
        "--prompt-max-peaks",
        type=int,
        default=30,
        help="Number of most intense MS2 peaks exposed to the LLM.",
    )

    parser.add_argument(
        "--output-rank-k",
        type=int,
        default=10,
        help=(
            "Number of candidates in the listwise assistant target. "
            "If a sample has fewer candidates, all available candidates are output."
        ),
    )

    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--sample-timeout", type=int, default=300)
    parser.add_argument("--progress-every", type=int, default=20)

    parser.add_argument(
        "--max-scan",
        type=int,
        default=0,
        help="0 = scan as many shuffled train records as needed.",
    )

    parser.add_argument(
        "--output-jsonl",
        default="./outputs/formula_nothink_sft_v3_listwise_zh_train10.jsonl",
    )

    parser.add_argument(
        "--audit-jsonl",
        default="./outputs/formula_nothink_sft_v3_listwise_zh_train10_audit.jsonl",
    )

    parser.add_argument(
        "--stats-json",
        default="./outputs/formula_nothink_sft_v3_listwise_zh_train10_stats.json",
    )

    parser.add_argument(
        "--state-jsonl",
        default="./outputs/formula_nothink_sft_v4_resume_state.jsonl",
        help=(
            "Streaming resume state. One compact record is written for every "
            "scanned training spectrum."
        ),
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from --state-jsonl and append to existing training/audit "
            "JSONL files. Already scanned records are not recomputed."
        ),
    )

    args = parser.parse_args()

    if args.target_examples <= 0:
        raise ValueError("--target-examples must be > 0")

    if not (0 <= args.rare_frac <= 1):
        raise ValueError("--rare-frac must be in [0,1]")

    if not (0 <= args.hard_frac <= 1):
        raise ValueError("--hard-frac must be in [0,1]")

    quotas = compute_quotas(
        args.target_examples,
        args.rare_frac,
        args.hard_frac,
    )

    base = import_module_from_file(
        "formula_sft_base_parent",
        args.base_file,
    )

    _ = base.load_rule_module(args.rule_file)

    all_records = base.load_records(args.input)

    signatures, signature_counts = base.build_extension_signatures(
        all_records,
        args.extension_min_train_count,
        args.max_extension_signatures,
    )

    records = [
        record
        for record in all_records
        if str(base.get_fold(record)).lower() == args.fold.lower()
    ]

    total_fold_records = len(records)

    if total_fold_records == 0:
        raise RuntimeError(
            f"No records found for fold={args.fold!r}"
        )

    # Reproducible record order before expensive candidate generation.
    rng = random.Random(args.seed)

    indexed_records = list(enumerate(records, 1))
    rng.shuffle(indexed_records)

    if args.max_scan > 0:
        indexed_records = indexed_records[:args.max_scan]

    limits = base.parse_limits(args.max_counts)

    print("=" * 104)
    print("BUILD NO-THINK LISTWISE FORMULA-RERANKING SFT DATA")
    print("=" * 104)
    print(f"Fold records                  : {total_fold_records}")
    print(f"Target examples               : {args.target_examples}")
    print(f"Sampling quotas               : {quotas}")
    print()
    print("LOCKED CANDIDATE POOL")
    print(f"  V3 global                   : Top-{args.v3_short}")
    print(f"  OLD                         : Top-{args.old_short}")
    print(f"  DIRECT                      : Top-{args.direct_short}")
    print(f"  V3-F reserved               : Top-{args.f_short}")
    print("  Neutral-formula dedup       : ON")
    print("  SIRIUS rescue               : OFF")
    print()
    print(f"Prompt MS2 peaks              : Top-{args.prompt_max_peaks} intense peaks")
    print(f"Listwise target               : Top-{args.output_rank_k}")
    print(f"Candidate shuffle             : {args.shuffle_candidates}")
    print(f"Extension signatures          : {len(signatures)}")
    print(f"Workers                       : {args.num_workers}")
    print(f"Sample timeout                : {args.sample_timeout}s")
    print("Teacher model                 : OFF")
    print("Chain-of-thought              : OFF")
    print("=" * 104, flush=True)

    output_path = Path(args.output_jsonl)
    audit_path = Path(args.audit_jsonl)
    stats_path = Path(args.stats_json)
    state_path = Path(args.state_jsonl)

    for path in (output_path, audit_path, stats_path, state_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    accepted_by_category = Counter()
    counters = Counter()
    processed_indices = set()
    candidate_counts: List[int] = []
    peak_counts: List[int] = []

    # ---------------------------------------------------------------------
    # Resume state
    # ---------------------------------------------------------------------
    if args.resume:
        if not state_path.exists():
            raise FileNotFoundError(
                f"--resume was requested but state file does not exist: {state_path}"
            )

        with state_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()

                if not line:
                    continue

                try:
                    state = json.loads(line)
                except Exception as exc:
                    raise RuntimeError(
                        f"Invalid resume-state JSON at line {line_no}: {exc}"
                    ) from exc

                dataset_idx = int(state["dataset_idx"])
                processed_indices.add(dataset_idx)

                status = str(state.get("status", "unknown"))
                counters[status] += 1

                if state.get("quota_full_reject"):
                    counters["quota_full_reject"] += 1

                if state.get("accepted"):
                    category = str(state["category"])
                    accepted_by_category[category] += 1

                    if state.get("candidate_count") is not None:
                        candidate_counts.append(int(state["candidate_count"]))

                    if state.get("prompt_peak_count") is not None:
                        peak_counts.append(int(state["prompt_peak_count"]))

        accepted_count = sum(accepted_by_category.values())

        if accepted_count > 0:
            if not output_path.exists():
                raise FileNotFoundError(
                    f"Resume state contains {accepted_count} accepted examples, "
                    f"but training JSONL is missing: {output_path}"
                )

            if not audit_path.exists():
                raise FileNotFoundError(
                    f"Resume state contains {accepted_count} accepted examples, "
                    f"but audit JSONL is missing: {audit_path}"
                )

        print(
            f"Resume loaded                  : scanned={len(processed_indices)}, "
            f"accepted={accepted_count}, categories={dict(accepted_by_category)}",
            flush=True,
        )

    else:
        # Fresh run: intentionally reset all streaming files.
        output_path.write_text("", encoding="utf-8")
        audit_path.write_text("", encoding="utf-8")
        state_path.write_text("", encoding="utf-8")
        accepted_count = 0

    # Keep the original deterministic shuffled scan order, but do not recompute
    # anything already represented in the resume state.
    remaining_records = [
        payload
        for payload in indexed_records
        if int(payload[0]) not in processed_indices
    ]

    # If resuming a 10k run as a 30k run, the already accepted 10k examples
    # remain part of the larger dataset. New examples are appended until the
    # expanded quotas are satisfied.
    for category, count in accepted_by_category.items():
        if count > quotas.get(category, 0):
            raise RuntimeError(
                f"Existing resume data has {count} accepted {category} examples, "
                f"which exceeds the current quota {quotas.get(category, 0)}. "
                "Use a target size at least as large as the original run."
            )

    started = time.perf_counter()

    if accepted_count < args.target_examples and remaining_records:
        pool = mp.get_context("fork").Pool(
            processes=min(args.num_workers, max(1, len(remaining_records))),
            initializer=init_worker,
            initargs=(
                args.base_file,
                args.rule_file,
                vars(args).copy(),
                limits,
                signatures,
            ),
        )

        # Append mode is safe because every accepted sample and every scanned
        # sample is flushed immediately.
        train_f = output_path.open("a", encoding="utf-8")
        audit_f = audit_path.open("a", encoding="utf-8")
        state_f = state_path.open("a", encoding="utf-8")

        try:
            iterator = pool.imap(
                worker,
                remaining_records,
                chunksize=1,
            )

            for newly_scanned, result in enumerate(iterator, 1):
                status = result["status"]
                counters[status] += 1

                accepted = False
                quota_full_reject = False
                category = result.get("category")

                if status == "ok":
                    if accepted_by_category[category] < quotas[category]:
                        accepted = True
                        accepted_by_category[category] += 1
                        accepted_count += 1

                        train_f.write(
                            json.dumps(
                                result["train_record"],
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        train_f.flush()

                        audit_f.write(
                            json.dumps(
                                result["audit_record"],
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        audit_f.flush()

                        candidate_counts.append(
                            int(result["audit_record"]["candidate_count"])
                        )
                        peak_counts.append(
                            int(result["audit_record"]["prompt_peak_count"])
                        )
                    else:
                        quota_full_reject = True
                        counters["quota_full_reject"] += 1

                state_record = {
                    "dataset_idx": int(result["dataset_idx"]),
                    "status": status,
                    "category": category,
                    "accepted": accepted,
                    "quota_full_reject": quota_full_reject,
                    "candidate_count": (
                        int(result["audit_record"]["candidate_count"])
                        if accepted else None
                    ),
                    "prompt_peak_count": (
                        int(result["audit_record"]["prompt_peak_count"])
                        if accepted else None
                    ),
                }

                state_f.write(
                    json.dumps(state_record, ensure_ascii=False)
                    + "\n"
                )
                state_f.flush()

                processed_indices.add(int(result["dataset_idx"]))

                total_scanned = len(processed_indices)

                if (
                    args.progress_every > 0
                    and newly_scanned % args.progress_every == 0
                ):
                    elapsed = time.perf_counter() - started

                    print(
                        f"Scanned {total_scanned} | "
                        f"accepted={accepted_count}/{args.target_examples} "
                        f"{dict(accepted_by_category)} | "
                        f"gold_miss={counters['gold_miss']} "
                        f"timeout={counters['timeout']} "
                        f"error={counters['error']} | "
                        f"wall={elapsed/60:.1f} min",
                        file=sys.stderr,
                        flush=True,
                    )

                if accepted_count >= args.target_examples:
                    break

            pool.terminate()
            pool.join()

        except BaseException:
            # Streaming files have already been flushed. This makes Ctrl+C,
            # SSH loss, or machine interruption resumable.
            pool.terminate()
            pool.join()
            raise

        finally:
            train_f.close()
            audit_f.close()
            state_f.close()

    if accepted_count < args.target_examples:
        print(
            f"[WARN] only collected {accepted_count} / "
            f"{args.target_examples} requested examples. "
            f"Accepted categories: {dict(accepted_by_category)}",
            file=sys.stderr,
            flush=True,
        )

    stats = {
        "requested_examples": args.target_examples,
        "collected_examples": accepted_count,
        "sampling_quotas": quotas,
        "accepted_by_category": dict(accepted_by_category),
        "scan_counters": dict(counters),
        "scanned_unique_records": len(processed_indices),
        "resume_enabled": True,
        "seed": args.seed,

        "candidate_pool": {
            "v3_global_top": args.v3_short,
            "old_top": args.old_short,
            "direct_top": args.direct_short,
            "v3_f_top": args.f_short,
        },

        "candidate_count": {
            "mean": (
                sum(candidate_counts) / len(candidate_counts)
                if candidate_counts else None
            ),
            "min": min(candidate_counts) if candidate_counts else None,
            "max": max(candidate_counts) if candidate_counts else None,
        },

        "prompt_peak_count": {
            "mean": (
                sum(peak_counts) / len(peak_counts)
                if peak_counts else None
            ),
            "min": min(peak_counts) if peak_counts else None,
            "max": max(peak_counts) if peak_counts else None,
        },

        "training_format": "listwise_top10_nothink_v4_zh_resume",
        "output_rank_k": args.output_rank_k,
        "teacher_model": False,
        "chain_of_thought": False,
        "sirius_rescue": False,
    }

    stats_path.write_text(
        json.dumps(
            stats,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 104)
    print("DONE")
    print("=" * 104)
    print(f"Collected                    : {accepted_count}")
    print(f"Accepted categories          : {dict(accepted_by_category)}")
    print(f"Scanned unique records       : {len(processed_indices)}")

    if candidate_counts:
        print(
            f"Candidate count              : "
            f"mean={sum(candidate_counts)/len(candidate_counts):.2f}, "
            f"min={min(candidate_counts)}, "
            f"max={max(candidate_counts)}"
        )

    print(f"Training JSONL               : {output_path.resolve()}")
    print(f"Audit JSONL                  : {audit_path.resolve()}")
    print(f"Resume state                 : {state_path.resolve()}")
    print(f"Stats JSON                   : {stats_path.resolve()}")


if __name__ == "__main__":
    main()
