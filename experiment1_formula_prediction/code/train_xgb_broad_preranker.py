#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train an XGBoost learning-to-rank pre-ranker for broad formula candidates.

Input datasets are produced by build_broad_preranker_dataset.py.

Ranking unit:
- one spectrum = one query group
- one candidate formula = one row
- gold formula = label 1
- all other candidates = label 0

Default objective:
    rank:pairwise

Validation reporting:
- Practical-pool coverage
- Recall@1/3/5/10/20/30/50 on the FULL validation denominator
- Conditional Recall@K among groups whose gold exists in the practical pool
- Gold rank distribution among covered groups

This script does NOT call the LLM.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def load_dataset(prefix: str):
    prefix = Path(prefix)

    dataset_info = json.loads(
        Path(str(prefix) + ".dataset.json").read_text(
            encoding="utf-8"
        )
    )

    features_info = json.loads(
        Path(str(prefix) + ".features.json").read_text(
            encoding="utf-8"
        )
    )

    feature_names = features_info["feature_names"]
    n_rows = int(dataset_info["n_rows"])
    feature_dim = int(dataset_info["feature_dim"])

    if feature_dim != len(feature_names):
        raise RuntimeError(
            "feature_dim mismatch"
        )

    X_path = Path(str(prefix) + ".X.f32.bin")
    y_path = Path(str(prefix) + ".y.u8.bin")
    groups_path = Path(str(prefix) + ".groups.i32.bin")

    X = np.memmap(
        X_path,
        dtype=np.float32,
        mode="r",
        shape=(n_rows, feature_dim),
    )

    y = np.memmap(
        y_path,
        dtype=np.uint8,
        mode="r",
        shape=(n_rows,),
    )

    groups = np.fromfile(
        groups_path,
        dtype=np.int32,
    )

    if int(groups.sum()) != n_rows:
        raise RuntimeError(
            f"group rows mismatch: {groups.sum()} vs {n_rows}"
        )

    return {
        "prefix": str(prefix),
        "info": dataset_info,
        "feature_names": feature_names,
        "X": X,
        "y": y,
        "groups": groups,
    }


def group_offsets(groups: np.ndarray):
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


def subset_positive_groups(
    X,
    y,
    groups,
):
    starts, ends = group_offsets(groups)

    positive_group_ids = []

    for i, (s, e) in enumerate(
        zip(starts, ends)
    ):
        if int(np.asarray(y[s:e]).sum()) > 0:
            positive_group_ids.append(i)

    if not positive_group_ids:
        raise RuntimeError(
            "No positive validation groups"
        )

    X_parts = []
    y_parts = []
    g_parts = []

    for i in positive_group_ids:
        s, e = int(starts[i]), int(ends[i])
        X_parts.append(
            np.asarray(X[s:e], dtype=np.float32)
        )
        y_parts.append(
            np.asarray(y[s:e], dtype=np.float32)
        )
        g_parts.append(e - s)

    X_pos = np.concatenate(
        X_parts,
        axis=0,
    )
    y_pos = np.concatenate(
        y_parts,
        axis=0,
    )
    groups_pos = np.asarray(
        g_parts,
        dtype=np.int32,
    )

    return X_pos, y_pos, groups_pos


def evaluate_recall(
    scores: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    ks: List[int],
):
    starts, ends = group_offsets(
        groups
    )

    totals = {
        k: 0
        for k in ks
    }
    conditional = {
        k: 0
        for k in ks
    }

    covered = 0
    gold_ranks = []

    for s, e in zip(starts, ends):
        s, e = int(s), int(e)

        yy = np.asarray(
            y[s:e],
            dtype=np.uint8,
        )
        ss = np.asarray(
            scores[s:e],
            dtype=np.float64,
        )

        pos = np.flatnonzero(
            yy > 0
        )

        if len(pos) == 0:
            continue

        covered += 1

        # Exactly one gold is expected, but robust to >1.
        order = np.argsort(
            -ss,
            kind="mergesort",
        )

        best_gold_rank = min(
            int(np.where(order == p)[0][0]) + 1
            for p in pos
        )

        gold_ranks.append(
            best_gold_rank
        )

        for k in ks:
            hit = int(
                best_gold_rank <= k
            )
            totals[k] += hit
            conditional[k] += hit

    n_groups = len(groups)

    full_recall = {
        k: totals[k] / n_groups
        if n_groups else 0.0
        for k in ks
    }

    conditional_recall = {
        k: conditional[k] / covered
        if covered else 0.0
        for k in ks
    }

    return {
        "n_groups": n_groups,
        "covered_groups": covered,
        "pool_coverage": (
            covered / n_groups
            if n_groups else 0.0
        ),
        "full_recall": full_recall,
        "conditional_recall": conditional_recall,
        "gold_ranks": gold_ranks,
    }


def percentile(values, q):
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])

    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)

    if lo == hi:
        return float(xs[lo])

    return float(
        xs[lo]
        + (xs[hi] - xs[lo])
        * (pos - lo)
    )


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--train-prefix",
        required=True,
    )
    p.add_argument(
        "--val-prefix",
        required=True,
    )

    p.add_argument(
        "--output-model",
        required=True,
    )
    p.add_argument(
        "--output-summary",
        required=True,
    )
    p.add_argument(
        "--output-importance",
        required=True,
    )
    p.add_argument(
        "--output-val-scores",
        required=True,
    )

    p.add_argument(
        "--objective",
        default="rank:pairwise",
    )
    p.add_argument(
        "--eval-metric",
        default="ndcg@30",
    )

    p.add_argument(
        "--num-boost-round",
        type=int,
        default=800,
    )
    p.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=50,
    )

    p.add_argument(
        "--eta",
        type=float,
        default=0.05,
    )
    p.add_argument(
        "--max-depth",
        type=int,
        default=8,
    )
    p.add_argument(
        "--min-child-weight",
        type=float,
        default=5.0,
    )
    p.add_argument(
        "--subsample",
        type=float,
        default=0.85,
    )
    p.add_argument(
        "--colsample-bytree",
        type=float,
        default=0.90,
    )
    p.add_argument(
        "--reg-lambda",
        type=float,
        default=5.0,
    )
    p.add_argument(
        "--reg-alpha",
        type=float,
        default=0.0,
    )

    p.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
    )
    p.add_argument(
        "--nthread",
        type=int,
        default=16,
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = p.parse_args()

    import xgboost as xgb

    print(
        "xgboost version:",
        xgb.__version__,
    )

    train = load_dataset(
        args.train_prefix
    )
    val = load_dataset(
        args.val_prefix
    )

    if (
        train["feature_names"]
        != val["feature_names"]
    ):
        raise RuntimeError(
            "Train/val feature names differ."
        )

    feature_names = train[
        "feature_names"
    ]

    # XGBoost forbids [, ] and < in feature names.
    # Keep dataset files unchanged; sanitize names only for DMatrix/model.
    feature_names = [
        str(name)
        .replace("[", "_")
        .replace("]", "_")
        .replace("<", "_")
        for name in feature_names
    ]

    print("=" * 108)
    print("XGBOOST BROAD PRE-RANKER")
    print("=" * 108)
    print(
        "Train groups      :",
        len(train["groups"]),
    )
    print(
        "Train rows        :",
        train["info"]["n_rows"],
    )
    print(
        "Validation groups :",
        len(val["groups"]),
    )
    print(
        "Validation rows   :",
        val["info"]["n_rows"],
    )
    print(
        "Feature dim       :",
        len(feature_names),
    )
    print(
        "Device            :",
        args.device,
    )
    print("=" * 108)

    # Train DMatrix.
    dtrain = xgb.DMatrix(
        train["X"],
        label=np.asarray(
            train["y"],
            dtype=np.float32,
        ),
        feature_names=feature_names,
        nthread=args.nthread,
    )
    dtrain.set_group(
        train["groups"]
    )

    # Early stopping should use only validation groups that actually
    # contain a positive candidate.
    X_val_pos, y_val_pos, groups_val_pos = (
        subset_positive_groups(
            val["X"],
            val["y"],
            val["groups"],
        )
    )

    dval_pos = xgb.DMatrix(
        X_val_pos,
        label=y_val_pos,
        feature_names=feature_names,
        nthread=args.nthread,
    )
    dval_pos.set_group(
        groups_val_pos
    )

    params = {
        "objective": args.objective,
        "eval_metric": args.eval_metric,
        "eta": args.eta,
        "max_depth": args.max_depth,
        "min_child_weight": args.min_child_weight,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "lambda": args.reg_lambda,
        "alpha": args.reg_alpha,
        "tree_method": "hist",
        "nthread": args.nthread,
        "seed": args.seed,
        "verbosity": 1,
    }

    # XGBoost >=2 supports device=cuda. Older versions use gpu_hist.
    major = int(
        str(xgb.__version__).split(".")[0]
    )

    if args.device == "cuda":
        if major >= 2:
            params["device"] = "cuda"
        else:
            params["tree_method"] = "gpu_hist"

    evals_result = {}

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=args.num_boost_round,
        evals=[
            (dtrain, "train"),
            (dval_pos, "val_gold"),
        ],
        early_stopping_rounds=(
            args.early_stopping_rounds
        ),
        evals_result=evals_result,
        verbose_eval=25,
    )

    model_path = Path(
        args.output_model
    )
    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    booster.save_model(
        str(model_path)
    )

    # Full validation prediction.
    dval_all = xgb.DMatrix(
        val["X"],
        feature_names=feature_names,
        nthread=args.nthread,
    )

    scores = booster.predict(
        dval_all
    ).astype(
        np.float32,
        copy=False,
    )

    scores_path = Path(
        args.output_val_scores
    )
    scores_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    scores.tofile(
        scores_path
    )

    ks = [
        1,
        3,
        5,
        10,
        20,
        30,
        50,
    ]

    metrics = evaluate_recall(
        scores,
        val["y"],
        val["groups"],
        ks,
    )

    print("\n" + "=" * 108)
    print("VALIDATION CANDIDATE RECALL")
    print("=" * 108)
    print(
        f"Practical-pool coverage : "
        f"{metrics['covered_groups']}/{metrics['n_groups']} "
        f"= {100*metrics['pool_coverage']:.2f}%"
    )

    print("\n|K|Recall@K full denominator|Conditional Recall@K among covered|")
    print("|---:|---:|---:|")

    for k in ks:
        print(
            f"|{k}|"
            f"{100*metrics['full_recall'][k]:.2f}%|"
            f"{100*metrics['conditional_recall'][k]:.2f}%|"
        )

    gold_ranks = metrics[
        "gold_ranks"
    ]

    if gold_ranks:
        print("\nGold rank among covered groups:")
        print(
            f"  mean   : "
            f"{statistics.mean(gold_ranks):.2f}"
        )
        print(
            f"  median : "
            f"{statistics.median(gold_ranks):.1f}"
        )
        print(
            f"  P90    : "
            f"{percentile(gold_ranks,0.90):.1f}"
        )
        print(
            f"  P95    : "
            f"{percentile(gold_ranks,0.95):.1f}"
        )
        print(
            f"  max    : "
            f"{max(gold_ranks)}"
        )

    # Feature importance.
    importance = booster.get_score(
        importance_type="gain"
    )

    importance_rows = [
        {
            "feature": name,
            "gain": float(
                importance.get(
                    name,
                    0.0,
                )
            ),
        }
        for name in feature_names
    ]

    importance_rows.sort(
        key=lambda x: -x["gain"]
    )

    importance_path = Path(
        args.output_importance
    )
    importance_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with importance_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "feature",
                "gain",
            ],
        )
        writer.writeheader()
        writer.writerows(
            importance_rows
        )

    summary = {
        "xgboost_version": xgb.__version__,
        "best_iteration": getattr(
            booster,
            "best_iteration",
            None,
        ),
        "best_score": getattr(
            booster,
            "best_score",
            None,
        ),
        "params": params,
        "train": {
            "groups": int(
                len(train["groups"])
            ),
            "rows": int(
                train["info"]["n_rows"]
            ),
        },
        "validation": {
            "groups": int(
                metrics["n_groups"]
            ),
            "rows": int(
                val["info"]["n_rows"]
            ),
            "practical_pool_coverage": (
                metrics["pool_coverage"]
            ),
            "full_recall": {
                str(k): metrics[
                    "full_recall"
                ][k]
                for k in ks
            },
            "conditional_recall": {
                str(k): metrics[
                    "conditional_recall"
                ][k]
                for k in ks
            },
            "gold_rank": {
                "mean": (
                    statistics.mean(
                        gold_ranks
                    )
                    if gold_ranks
                    else None
                ),
                "median": (
                    statistics.median(
                        gold_ranks
                    )
                    if gold_ranks
                    else None
                ),
                "p90": percentile(
                    gold_ranks,
                    0.90,
                ),
                "p95": percentile(
                    gold_ranks,
                    0.95,
                ),
                "max": (
                    max(gold_ranks)
                    if gold_ranks
                    else None
                ),
            },
        },
        "top_features_by_gain": (
            importance_rows[:20]
        ),
        "paths": {
            "model": str(
                model_path.resolve()
            ),
            "val_scores": str(
                scores_path.resolve()
            ),
            "importance": str(
                importance_path.resolve()
            ),
        },
    }

    summary_path = Path(
        args.output_summary
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

    print("\nSaved:")
    print(
        " model     :",
        model_path.resolve(),
    )
    print(
        " summary   :",
        summary_path.resolve(),
    )
    print(
        " importance:",
        importance_path.resolve(),
    )
    print(
        " val scores:",
        scores_path.resolve(),
    )


if __name__ == "__main__":
    main()
