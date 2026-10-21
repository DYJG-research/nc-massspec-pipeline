#!/usr/bin/env python3
"""Compute MassSpecGym-compatible MCES@1 and MCES@10 from JSON/JSONL predictions.

The implementation follows MassSpecGym's de novo evaluation convention:
  * myopic-mces threshold = 15
  * always_stronger_bound = True
  * invalid or missing predictions receive distance 100
  * each per-sample distance is capped at 100
  * MCES@k is the mean, over samples, of the minimum distance among the first k predictions

Expected pipeline fields are supported directly:
  truth JSONL: identifier + smiles/true_smiles
  prediction JSONL: identifier + predictions (list of SMILES)
The loader also accepts several common alternative key names.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.error")

try:
    import pulp
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency 'pulp'. Install with:\n"
        "  python -m pip install 'pulp==2.9.0' 'myopic-mces==1.0.0'"
    ) from exc

try:
    # MassSpecGym 1.3.1 / myopic-mces 1.0.0 import path.
    from myopic_mces.myopic_mces import MCES
except ImportError:
    try:
        # Compatibility with newer myopic-mces releases.
        from myopic_mces import MCES  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "Missing dependency 'myopic-mces'. Install with:\n"
            "  python -m pip install 'pulp==2.9.0' 'myopic-mces==1.0.0'"
        ) from exc

ID_KEYS = (
    "identifier",
    "id",
    "spectrum_id",
    "spectrumid",
    "name",
)
TRUE_SMILES_KEYS = (
    "true_smiles",
    "ground_truth_smiles",
    "groundtruth_smiles",
    "target_smiles",
    "reference_smiles",
    "smiles",
)
PREDICTION_KEYS = (
    "predictions",
    "mols_pred",
    "top10",
    "top_10",
    "predicted_smiles",
    "smiles_predictions",
)
NESTED_KEYS = (
    "metadata",
    "molecule",
    "mol",
    "truth",
    "target",
    "reference",
    "prediction",
    "result",
    "output",
)

_WORKER_SOLVER: str | None = None
_WORKER_THRESHOLD = 15
_WORKER_ALWAYS_STRONGER = True


def normalize_key(value: str) -> str:
    return "".join(ch.lower() for ch in value if ch.isalnum())


def iter_nested_dicts(row: dict[str, Any], max_depth: int = 4) -> Iterable[dict[str, Any]]:
    """Yield a row and likely nested dictionaries without recursively scanning huge arrays."""
    stack: list[tuple[dict[str, Any], int]] = [(row, 0)]
    seen: set[int] = set()
    wanted_nested = {normalize_key(k) for k in NESTED_KEYS}
    while stack:
        current, depth = stack.pop()
        obj_id = id(current)
        if obj_id in seen:
            continue
        seen.add(obj_id)
        yield current
        if depth >= max_depth:
            continue
        for key, value in current.items():
            if isinstance(value, dict) and normalize_key(str(key)) in wanted_nested:
                stack.append((value, depth + 1))


def get_first_value(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    wanted = {normalize_key(k) for k in keys}
    for obj in iter_nested_dicts(row):
        for key, value in obj.items():
            if normalize_key(str(key)) in wanted and value not in (None, ""):
                return value
    return None


def get_identifier(row: dict[str, Any]) -> str:
    value = get_first_value(row, ID_KEYS)
    if value is None:
        raise KeyError("No identifier field found")
    return str(value).strip()


def load_records(path: Path) -> list[dict[str, Any]]:
    """Load JSONL, a JSON list, or a JSON dictionary of records."""
    if not path.is_file():
        raise FileNotFoundError(path)

    text = path.read_text(encoding="utf-8-sig")
    stripped = text.lstrip()
    if not stripped:
        return []

    if stripped[0] in "[{":
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, list):
            if not all(isinstance(row, dict) for row in obj):
                raise ValueError(f"{path}: JSON list contains non-object values")
            return obj
        if isinstance(obj, dict):
            for key in ("records", "data", "rows", "predictions", "results"):
                value = obj.get(key)
                if isinstance(value, list) and all(isinstance(row, dict) for row in value):
                    return value
            # Dictionary keyed by identifier.
            rows: list[dict[str, Any]] = []
            for key, value in obj.items():
                if isinstance(value, dict):
                    row = dict(value)
                    row.setdefault("identifier", key)
                    rows.append(row)
            if rows:
                return rows

    rows = []
    for line_no, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSONL") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_no}: expected a JSON object")
        rows.append(row)
    return rows


def extract_true_smiles(row: dict[str, Any]) -> str | None:
    value = get_first_value(row, TRUE_SMILES_KEYS)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _append_smiles(values: list[str], value: Any) -> None:
    if isinstance(value, str):
        value = value.strip()
        if value:
            values.append(value)
    elif isinstance(value, dict):
        for key in ("smiles", "SMILES", "canonical_smiles", "predicted_smiles"):
            item = value.get(key)
            if isinstance(item, str) and item.strip():
                values.append(item.strip())
                break


def extract_predictions(row: dict[str, Any]) -> list[str]:
    values: list[str] = []

    direct = get_first_value(row, PREDICTION_KEYS)
    if isinstance(direct, list):
        for item in direct:
            _append_smiles(values, item)
    elif isinstance(direct, str):
        # A single SMILES is accepted as Top-1.
        _append_smiles(values, direct)

    if not values:
        candidates = row.get("candidates")
        if isinstance(candidates, list):
            for item in candidates:
                _append_smiles(values, item)

    # Preserve ranking and duplicates exactly as supplied by the prediction file.
    return values


def validate_smiles(smiles: str | None) -> bool:
    return bool(smiles) and Chem.MolFromSmiles(smiles) is not None


def init_worker(solver: str, threshold: int, always_stronger_bound: bool) -> None:
    global _WORKER_SOLVER, _WORKER_THRESHOLD, _WORKER_ALWAYS_STRONGER
    _WORKER_SOLVER = solver
    _WORKER_THRESHOLD = threshold
    _WORKER_ALWAYS_STRONGER = always_stronger_bound


def parse_mces_result(result: Any) -> tuple[float, str | None]:
    """Return distance and an optional computation-status string."""
    if isinstance(result, (tuple, list)):
        if len(result) < 2:
            raise ValueError(f"Unexpected MCES result: {result!r}")
        distance = float(result[1])
        status = str(result[3]) if len(result) > 3 else None
        return distance, status
    if isinstance(result, (int, float)):
        return float(result), None
    raise TypeError(f"Unexpected MCES return type: {type(result).__name__}")


def compute_pair(task: tuple[str, str]) -> tuple[tuple[str, str], float, str | None, str | None, float]:
    true_smiles, pred_smiles = task
    started = time.perf_counter()
    try:
        result = MCES(
            s1=true_smiles,
            s2=pred_smiles,
            ind=0,
            threshold=_WORKER_THRESHOLD,
            always_stronger_bound=_WORKER_ALWAYS_STRONGER,
            solver=_WORKER_SOLVER,
            solver_options={"msg": 0},
        )
        distance, status = parse_mces_result(result)
        if not math.isfinite(distance):
            raise ValueError(f"non-finite distance: {distance}")
        return task, distance, status, None, time.perf_counter() - started
    except Exception as exc:  # keep the full evaluation running
        return task, 100.0, None, repr(exc), time.perf_counter() - started


def choose_solver(requested: str | None) -> str:
    available = pulp.listSolvers(onlyAvailable=True)
    if not available:
        raise RuntimeError(
            "No PuLP solver is available. With the pinned versions, reinstall using:\n"
            "  python -m pip install --force-reinstall 'pulp==2.9.0'\n"
            "Then check: python -c \"import pulp; print(pulp.listSolvers(onlyAvailable=True))\""
        )
    if requested:
        if requested not in available:
            raise RuntimeError(
                f"Requested solver {requested!r} is unavailable. Available: {available}"
            )
        return requested
    if "PULP_CBC_CMD" in available:
        return "PULP_CBC_CMD"
    return available[0]


def mean_or_nan(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else float("nan")


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * q
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return float(ordered[low])
    weight = pos - low
    return float(ordered[low] * (1 - weight) + ordered[high] * weight)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute MassSpecGym-compatible MCES@1 and MCES@10."
    )
    parser.add_argument("--truth", required=True, help="Truth JSON/JSONL file")
    parser.add_argument("--predictions", required=True, help="Prediction JSON/JSONL file")
    parser.add_argument("--output", required=True, help="Summary JSON output")
    parser.add_argument("--details", help="Optional per-sample JSONL output")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--solver", default=None, help="PuLP solver name; auto-select by default")
    parser.add_argument("--threshold", type=int, default=15)
    parser.add_argument("--invalid-distance", type=float, default=100.0)
    parser.add_argument("--cap", type=float, default=100.0)
    parser.add_argument(
        "--scope",
        choices=("truth", "intersection"),
        default="truth",
        help=(
            "truth: score every truth row and penalize missing predictions; "
            "intersection: score only identifiers present in both files"
        ),
    )
    parser.add_argument(
        "--dynamic-weaker-bound",
        action="store_true",
        help="Use the faster but non-official weaker-bound mode",
    )
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be at least 1")

    truth_path = Path(args.truth)
    prediction_path = Path(args.predictions)
    output_path = Path(args.output)
    details_path = Path(args.details) if args.details else output_path.with_suffix(".details.jsonl")

    truth_rows = load_records(truth_path)
    prediction_rows = load_records(prediction_path)

    truth_map: dict[str, str] = {}
    truth_duplicates: list[str] = []
    invalid_truth_ids: list[str] = []
    for row in truth_rows:
        sid = get_identifier(row)
        smiles = extract_true_smiles(row)
        if sid in truth_map:
            truth_duplicates.append(sid)
        if not validate_smiles(smiles):
            invalid_truth_ids.append(sid)
            continue
        truth_map[sid] = str(smiles)

    prediction_map: dict[str, list[str]] = {}
    prediction_duplicates: list[str] = []
    for row in prediction_rows:
        sid = get_identifier(row)
        if sid in prediction_map:
            prediction_duplicates.append(sid)
        prediction_map[sid] = extract_predictions(row)

    if not truth_map:
        raise RuntimeError(f"No valid truth SMILES found in {truth_path}")

    if args.scope == "intersection":
        identifiers = sorted(set(truth_map) & set(prediction_map))
    else:
        identifiers = sorted(truth_map)

    if not identifiers:
        raise RuntimeError("No identifiers to evaluate after applying --scope")

    solver = choose_solver(args.solver)
    always_stronger = not args.dynamic_weaker_bound

    # Build the unique valid MCES pair set required for Top-10 evaluation.
    pairs: set[tuple[str, str]] = set()
    invalid_prediction_count = 0
    missing_prediction_ids: list[str] = []
    for sid in identifiers:
        preds = prediction_map.get(sid, [])[:10]
        if not preds:
            missing_prediction_ids.append(sid)
        for pred in preds:
            if validate_smiles(pred):
                pairs.add((truth_map[sid], pred))
            else:
                invalid_prediction_count += 1

    print(f"[INFO] truth rows:              {len(truth_rows)}", flush=True)
    print(f"[INFO] prediction rows:         {len(prediction_rows)}", flush=True)
    print(f"[INFO] evaluated identifiers:   {len(identifiers)}", flush=True)
    print(f"[INFO] unique valid MCES pairs: {len(pairs)}", flush=True)
    print(f"[INFO] solver:                  {solver}", flush=True)
    print(f"[INFO] threshold:               {args.threshold}", flush=True)
    print(f"[INFO] always_stronger_bound:   {always_stronger}", flush=True)

    pair_cache: dict[tuple[str, str], float] = {}
    pair_status: dict[tuple[str, str], str | None] = {}
    pair_errors: dict[tuple[str, str], str] = {}
    pair_seconds: dict[tuple[str, str], float] = {}

    sorted_pairs = sorted(pairs)
    if sorted_pairs:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=init_worker,
            initargs=(solver, args.threshold, always_stronger),
        ) as executor:
            futures = {executor.submit(compute_pair, pair): pair for pair in sorted_pairs}
            total = len(futures)
            for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
                pair, distance, status, error, seconds = future.result()
                pair_cache[pair] = min(float(distance), args.cap)
                pair_status[pair] = status
                pair_seconds[pair] = seconds
                if error:
                    pair_errors[pair] = error
                if completed == 1 or completed % 25 == 0 or completed == total:
                    print(f"[MCES] {completed}/{total}", flush=True)

    details: list[dict[str, Any]] = []
    top1_values: list[float] = []
    top10_values: list[float] = []

    for sid in identifiers:
        true_smiles = truth_map[sid]
        predictions = prediction_map.get(sid, [])[:10]
        candidate_rows: list[dict[str, Any]] = []

        for rank, pred in enumerate(predictions, 1):
            if validate_smiles(pred):
                pair = (true_smiles, pred)
                distance = pair_cache.get(pair, args.invalid_distance)
                error = pair_errors.get(pair)
                status = pair_status.get(pair)
            else:
                distance = args.invalid_distance
                error = "invalid_smiles"
                status = None
            distance = min(float(distance), args.cap)
            candidate_rows.append(
                {
                    "rank": rank,
                    "smiles": pred,
                    "mces": distance,
                    "mces_status": status,
                    "error": error,
                }
            )

        if candidate_rows:
            top1 = candidate_rows[0]["mces"]
            best_top10 = min(candidate_rows, key=lambda item: item["mces"])
            top10 = best_top10["mces"]
            best_rank = best_top10["rank"]
            best_smiles = best_top10["smiles"]
        else:
            top1 = min(args.invalid_distance, args.cap)
            top10 = min(args.invalid_distance, args.cap)
            best_rank = None
            best_smiles = None

        top1_values.append(float(top1))
        top10_values.append(float(top10))
        details.append(
            {
                "identifier": sid,
                "true_smiles": true_smiles,
                "num_predictions": len(predictions),
                "top1_smiles": predictions[0] if predictions else None,
                "mces_at_1": float(top1),
                "best_top10_rank": best_rank,
                "best_top10_smiles": best_smiles,
                "mces_at_10": float(top10),
                "candidates": candidate_rows,
            }
        )

    summary = {
        "metric": "MassSpecGym-compatible myopic MCES",
        "truth": str(truth_path),
        "predictions": str(prediction_path),
        "scope": args.scope,
        "num_truth_rows": len(truth_rows),
        "num_valid_truth": len(truth_map),
        "num_prediction_rows": len(prediction_rows),
        "num_evaluated": len(identifiers),
        "num_missing_prediction_rows": len(missing_prediction_ids),
        "num_invalid_prediction_smiles_top10": invalid_prediction_count,
        "num_unique_mces_pairs": len(pairs),
        "num_mces_pair_errors": len(pair_errors),
        "mces_at_1": mean_or_nan(top1_values),
        "mces_at_10": mean_or_nan(top10_values),
        "mces_at_1_median": percentile(top1_values, 0.5),
        "mces_at_10_median": percentile(top10_values, 0.5),
        "mces_at_1_q25": percentile(top1_values, 0.25),
        "mces_at_1_q75": percentile(top1_values, 0.75),
        "mces_at_10_q25": percentile(top10_values, 0.25),
        "mces_at_10_q75": percentile(top10_values, 0.75),
        "settings": {
            "myopic_mces_version_target": "1.0.0",
            "pulp_version_target": "2.9.0",
            "solver": solver,
            "threshold": args.threshold,
            "always_stronger_bound": always_stronger,
            "invalid_or_missing_distance": args.invalid_distance,
            "per_sample_cap": args.cap,
            "top_ks": [1, 10],
        },
        "diagnostics": {
            "invalid_truth_ids": invalid_truth_ids,
            "duplicate_truth_ids": sorted(set(truth_duplicates)),
            "duplicate_prediction_ids": sorted(set(prediction_duplicates)),
            "missing_prediction_ids": missing_prediction_ids,
            "prediction_only_ids": sorted(set(prediction_map) - set(truth_map)),
            "mces_pair_errors": [
                {
                    "true_smiles": pair[0],
                    "predicted_smiles": pair[1],
                    "error": error,
                }
                for pair, error in pair_errors.items()
            ],
            "total_pair_compute_seconds": sum(pair_seconds.values()),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    details_path.parent.mkdir(parents=True, exist_ok=True)
    with details_path.open("w", encoding="utf-8") as f:
        for row in details:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\n================ MCES SUMMARY ================", flush=True)
    print(f"Samples:  {summary['num_evaluated']}", flush=True)
    print(f"MCES@1:   {summary['mces_at_1']:.6f}", flush=True)
    print(f"MCES@10:  {summary['mces_at_10']:.6f}", flush=True)
    print(f"Missing:  {summary['num_missing_prediction_rows']}", flush=True)
    print(f"Errors:   {summary['num_mces_pair_errors']}", flush=True)
    print(f"Summary:  {output_path}", flush=True)
    print(f"Details:  {details_path}", flush=True)


if __name__ == "__main__":
    main()
