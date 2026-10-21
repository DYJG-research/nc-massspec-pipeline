#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Score a broad-preranker dataset and export a Top-K shortlist ready for Qwen.

Inputs:
- dataset prefix from build_broad_preranker_dataset.py
- trained XGBoost model

Outputs:
- shortlist JSONL, one spectrum per row
- summary JSON with shortlist Recall@K

Qwen rank-feature semantics are preserved:
- V3Rank only if candidate is in original V3 current Top10
- OldRank only if candidate is in original OLD Top10
- DirectRank only if candidate is in Direct Top5
- FRank only if candidate is in original F Top15
Deep candidates do not receive fake low ranks.
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np


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


def load_dataset(prefix: str):
    prefix = Path(prefix)

    info = json.loads(
        Path(str(prefix) + ".dataset.json").read_text(
            encoding="utf-8"
        )
    )

    features_info = json.loads(
        Path(str(prefix) + ".features.json").read_text(
            encoding="utf-8"
        )
    )

    feature_names = features_info[
        "feature_names"
    ]

    n_rows = int(
        info["n_rows"]
    )
    feature_dim = int(
        info["feature_dim"]
    )

    X = np.memmap(
        Path(str(prefix) + ".X.f32.bin"),
        dtype=np.float32,
        mode="r",
        shape=(n_rows, feature_dim),
    )

    y = np.memmap(
        Path(str(prefix) + ".y.u8.bin"),
        dtype=np.uint8,
        mode="r",
        shape=(n_rows,),
    )

    groups = np.fromfile(
        Path(str(prefix) + ".groups.i32.bin"),
        dtype=np.int32,
    )

    meta_path = Path(
        str(prefix) + ".meta.jsonl.gz"
    )

    return {
        "prefix": str(prefix),
        "info": info,
        "feature_names": feature_names,
        "X": X,
        "y": y,
        "groups": groups,
        "meta_path": meta_path,
    }


def read_meta(path: Path):
    with gzip.open(
        path,
        "rt",
        encoding="utf-8",
    ) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def group_offsets(groups):
    starts = np.zeros(
        len(groups),
        dtype=np.int64,
    )
    if len(groups) > 1:
        starts[1:] = np.cumsum(
            groups[:-1],
            dtype=np.int64,
        )
    ends = starts + groups.astype(
        np.int64
    )
    return starts, ends


def rank_map(base, formulas: Sequence[str]):
    out = {}
    for rank, formula in enumerate(
        formulas,
        1,
    ):
        key = base.formula_key(
            formula
        )
        if key is not None and key not in out:
            out[key] = rank
    return out


def build_qwen_candidate(
    base,
    builder,
    meta: Dict[str, Any],
    formula: str,
):
    key = base.formula_key(
        formula
    )

    v3_current = rank_map(
        base,
        meta.get(
            "v3_current_top10",
            [],
        ),
    )
    old_current = rank_map(
        base,
        meta.get(
            "old_current_top10",
            [],
        ),
    )
    direct_current = rank_map(
        base,
        meta.get(
            "direct_current_top5",
            [],
        ),
    )
    f_current = rank_map(
        base,
        meta.get(
            "f_current_top15",
            [],
        ),
    )

    v3_practical_set = {
        base.formula_key(x)
        for x in meta.get(
            "v3_practical",
            [],
        )
    }
    old_practical_set = {
        base.formula_key(x)
        for x in meta.get(
            "old_practical",
            [],
        )
    }
    direct_practical_set = {
        base.formula_key(x)
        for x in meta.get(
            "direct_practical",
            [],
        )
    }
    f_practical_set = {
        base.formula_key(x)
        for x in meta.get(
            "f_practical",
            [],
        )
    }

    vr = v3_current.get(
        key
    )
    orank = old_current.get(
        key
    )
    dr = direct_current.get(
        key
    )
    fr = f_current.get(
        key
    )

    # V3 and F count as one generator family.
    support = (
        int(
            key in v3_practical_set
            or key in f_practical_set
        )
        + int(
            key in old_practical_set
        )
        + int(
            key in direct_practical_set
        )
    )

    chemistry = builder.best_mass_matched_adduct(
        base,
        formula,
        float(meta["precursor_mz"]),
        meta["polarity"],
    )

    return {
        "formula": formula,
        "best_adduct": chemistry[
            "best_adduct"
        ],
        "ppm_error": chemistry[
            "ppm_error"
        ],
        "dbe": chemistry[
            "dbe"
        ],
        "v3_rank": vr,
        "old_rank": orank,
        "direct_rank": dr,
        "f_rank": fr,
        "support": support,
    }


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--dataset-prefix",
        required=True,
    )
    p.add_argument(
        "--model",
        required=True,
    )
    p.add_argument(
        "--base-file",
        required=True,
    )
    p.add_argument(
        "--builder-file",
        required=True,
    )

    p.add_argument(
        "--topk",
        type=int,
        default=30,
    )
    p.add_argument(
        "--nthread",
        type=int,
        default=16,
    )

    p.add_argument(
        "--output-jsonl",
        required=True,
    )
    p.add_argument(
        "--summary-json",
        required=True,
    )

    args = p.parse_args()

    import xgboost as xgb

    base = import_module_from_file(
        "shortlist_base",
        args.base_file,
    )
    builder = import_module_from_file(
        "shortlist_builder",
        args.builder_file,
    )

    ds = load_dataset(
        args.dataset_prefix
    )

    booster = xgb.Booster()
    booster.load_model(
        args.model
    )

    dmatrix = xgb.DMatrix(
        ds["X"],
        feature_names=[
            str(name)
            .replace("[", "_")
            .replace("]", "_")
            .replace("<", "_")
            for name in ds["feature_names"]
        ],
        nthread=args.nthread,
    )

    scores = booster.predict(
        dmatrix
    )

    starts, ends = group_offsets(
        ds["groups"]
    )

    metas = list(
        read_meta(
            ds["meta_path"]
        )
    )

    if len(metas) != len(
        ds["groups"]
    ):
        raise RuntimeError(
            f"Metadata/groups mismatch: "
            f"{len(metas)} vs {len(ds['groups'])}"
        )

    out_path = Path(
        args.output_jsonl
    )
    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_groups = len(
        ds["groups"]
    )
    pool_hits = 0
    shortlist_hits = 0
    candidate_counts = []

    with out_path.open(
        "w",
        encoding="utf-8",
    ) as fout:

        for group_idx, (
            meta,
            s,
            e,
        ) in enumerate(
            zip(
                metas,
                starts,
                ends,
            ),
            1,
        ):
            s, e = int(s), int(e)

            formulas = meta[
                "formulas"
            ]
            yy = np.asarray(
                ds["y"][s:e],
                dtype=np.uint8,
            )
            ss = np.asarray(
                scores[s:e],
                dtype=np.float64,
            )

            if len(formulas) != e - s:
                raise RuntimeError(
                    f"Formula count mismatch in "
                    f"{meta.get('identifier')}"
                )

            order = np.argsort(
                -ss,
                kind="mergesort",
            )

            top_idx = order[
                :min(
                    args.topk,
                    len(order),
                )
            ]

            top_formulas = [
                formulas[
                    int(i)
                ]
                for i in top_idx
            ]

            qwen_candidates = [
                build_qwen_candidate(
                    base,
                    builder,
                    meta,
                    formula,
                )
                for formula in top_formulas
            ]

            gold_in_pool = int(
                yy.sum() > 0
            )
            gold_in_shortlist = int(
                yy[top_idx].sum() > 0
            )

            pool_hits += gold_in_pool
            shortlist_hits += (
                gold_in_shortlist
            )
            candidate_counts.append(
                len(top_formulas)
            )

            row = {
                "group_idx": group_idx,
                "fold_idx": meta[
                    "fold_idx"
                ],
                "sample_order": meta[
                    "sample_order"
                ],
                "identifier": meta[
                    "identifier"
                ],
                "precursor_mz": meta[
                    "precursor_mz"
                ],
                "polarity": meta[
                    "polarity"
                ],
                "prompt_peaks": meta[
                    "prompt_peaks"
                ],

                # Evaluation-only.
                "true_formula": meta[
                    "true_formula"
                ],
                "true_adduct": meta[
                    "true_adduct"
                ],

                "practical_pool_count": meta[
                    "candidate_count"
                ],
                "gold_in_practical_pool": (
                    gold_in_pool
                ),
                "gold_in_shortlist": (
                    gold_in_shortlist
                ),

                "shortlist": (
                    qwen_candidates
                ),
                "preranker_scores": [
                    float(
                        ss[int(i)]
                    )
                    for i in top_idx
                ],
            }

            fout.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    summary = {
        "dataset_prefix": args.dataset_prefix,
        "model": str(
            Path(args.model).resolve()
        ),
        "topk": args.topk,
        "N": total_groups,
        "practical_pool_hits": (
            pool_hits
        ),
        "practical_pool_coverage": (
            pool_hits / total_groups
            if total_groups else 0.0
        ),
        "shortlist_hits": (
            shortlist_hits
        ),
        "shortlist_coverage": (
            shortlist_hits / total_groups
            if total_groups else 0.0
        ),
        "conditional_shortlist_recall": (
            shortlist_hits / pool_hits
            if pool_hits else 0.0
        ),
        "mean_shortlist_size": (
            statistics.mean(
                candidate_counts
            )
            if candidate_counts
            else None
        ),
    }

    summary_path = Path(
        args.summary_json
    )
    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 100)
    print("PRE-RANKER SHORTLIST EXPORT")
    print("=" * 100)
    print(
        f"N                       : "
        f"{total_groups}"
    )
    print(
        f"Practical pool coverage : "
        f"{pool_hits}/{total_groups} "
        f"= {100*pool_hits/total_groups if total_groups else 0:.2f}%"
    )
    print(
        f"Top-{args.topk} coverage         : "
        f"{shortlist_hits}/{total_groups} "
        f"= {100*shortlist_hits/total_groups if total_groups else 0:.2f}%"
    )
    print(
        f"Conditional retention    : "
        f"{100*shortlist_hits/pool_hits if pool_hits else 0:.2f}%"
    )
    print("\nSaved:")
    print(
        " shortlist:",
        out_path.resolve(),
    )
    print(
        " summary  :",
        summary_path.resolve(),
    )


if __name__ == "__main__":
    main()
