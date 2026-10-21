#!/usr/bin/env python3
"""
Evaluate Top-k molecular structure predictions with MassSpecGym-compatible metrics.

MCES compatibility target:
  myopic-mces == 1.0.0
  PuLP         == 2.9.0

The implementation follows the official MassSpecGym convention:
  - myopic MCES threshold = 15
  - always_stronger_bound = True
  - invalid/missing predictions receive MCES penalty 100
  - MCES@k is the minimum distance among the first k positions
  - the distance is retval[1] from myopic_mces.MCES(...)
  - any MCES computation exception aborts evaluation; errors are never replaced
    silently by a numeric score.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import sqlite3
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pulp
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


OFFICIAL_MYOPIC_MCES_VERSION = "1.0.0"
OFFICIAL_PULP_VERSION = "2.9.0"
INVALID_MCES_PENALTY = 100.0


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def mol_from_smiles(smiles: str | None) -> Chem.Mol | None:
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def canonical_smiles(smiles: str) -> str:
    mol = mol_from_smiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def inchi_key_2d_from_mol(mol: Chem.Mol) -> str:
    return Chem.MolToInchiKey(mol).split("-")[0]


def morgan_fp(mol: Chem.Mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)


def symmetric_pair_key(smiles_a: str, smiles_b: str) -> tuple[str, str, str]:
    a = canonical_smiles(smiles_a)
    b = canonical_smiles(smiles_b)
    left, right = sorted((a, b))
    digest = hashlib.sha256(
        (left + "\0" + right).encode("utf-8")
    ).hexdigest()
    return digest, left, right


def choose_solver() -> str:
    available = pulp.listSolvers(onlyAvailable=True)
    if not available:
        raise RuntimeError(
            "No PuLP solver is available. Install PuLP 2.9.0, which includes "
            "PULP_CBC_CMD in the official MassSpecGym environment."
        )
    if "PULP_CBC_CMD" in available:
        return "PULP_CBC_CMD"
    return available[0]


def mces_worker(task: tuple[str, str, str, int, str]) -> tuple[str, float | None, str | None]:
    pair_hash, smiles_a, smiles_b, threshold, solver = task
    try:
        from myopic_mces.myopic_mces import MCES

        retval = MCES(
            s1=smiles_a,
            s2=smiles_b,
            ind=0,
            threshold=threshold,
            always_stronger_bound=True,
            solver=solver,
            solver_options={"msg": 0},
        )

        if not isinstance(retval, (tuple, list)) or len(retval) < 2:
            raise RuntimeError(
                f"Unexpected MCES return value: {retval!r}"
            )

        distance = float(retval[1])
        if not math.isfinite(distance) or distance < 0:
            raise RuntimeError(
                f"Invalid MCES distance: {distance!r}; full return={retval!r}"
            )

        return pair_hash, distance, None
    except Exception as exc:
        return pair_hash, None, repr(exc)


class MCESCache:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS mces_cache (
                pair_hash TEXT PRIMARY KEY,
                smiles_a TEXT NOT NULL,
                smiles_b TEXT NOT NULL,
                threshold INTEGER NOT NULL,
                myopic_mces_version TEXT NOT NULL,
                pulp_version TEXT NOT NULL,
                distance REAL NOT NULL
            )
            """
        )
        self.conn.commit()

    def get_many(
        self,
        pair_hashes: list[str],
        threshold: int,
        myopic_version: str,
        pulp_version: str,
    ) -> dict[str, float]:
        if not pair_hashes:
            return {}
        result: dict[str, float] = {}
        chunk_size = 800
        for start in range(0, len(pair_hashes), chunk_size):
            chunk = pair_hashes[start : start + chunk_size]
            placeholders = ",".join("?" for _ in chunk)
            query = f"""
                SELECT pair_hash, distance
                FROM mces_cache
                WHERE pair_hash IN ({placeholders})
                  AND threshold = ?
                  AND myopic_mces_version = ?
                  AND pulp_version = ?
            """
            params: list[Any] = [
                *chunk,
                threshold,
                myopic_version,
                pulp_version,
            ]
            for pair_hash, distance in self.conn.execute(query, params):
                result[str(pair_hash)] = float(distance)
        return result

    def put_many(
        self,
        rows: list[tuple[str, str, str, int, str, str, float]],
    ) -> None:
        self.conn.executemany(
            """
            INSERT OR REPLACE INTO mces_cache (
                pair_hash,
                smiles_a,
                smiles_b,
                threshold,
                myopic_mces_version,
                pulp_version,
                distance
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--with-mces", action="store_true")
    parser.add_argument("--mces-workers", type=int, default=4)
    parser.add_argument("--mces-threshold", type=int, default=15)
    parser.add_argument(
        "--mces-cache",
        default=None,
        help="Optional SQLite cache shared across strategy evaluations.",
    )
    parser.add_argument(
        "--allow-version-mismatch",
        action="store_true",
        help="Not recommended. Allow versions different from official MassSpecGym.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    myopic_version = package_version("myopic-mces")
    pulp_version = package_version("PuLP")

    if args.with_mces and not args.allow_version_mismatch:
        mismatches = []
        if myopic_version != OFFICIAL_MYOPIC_MCES_VERSION:
            mismatches.append(
                f"myopic-mces={myopic_version}, expected {OFFICIAL_MYOPIC_MCES_VERSION}"
            )
        if pulp_version != OFFICIAL_PULP_VERSION:
            mismatches.append(
                f"PuLP={pulp_version}, expected {OFFICIAL_PULP_VERSION}"
            )
        if mismatches:
            raise RuntimeError(
                "MCES dependency mismatch:\n  "
                + "\n  ".join(mismatches)
                + "\nInstall the official MassSpecGym-compatible versions before evaluation."
            )

    prediction_path = Path(args.predictions)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        json.loads(line)
        for line in prediction_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    details: list[dict[str, Any]] = []
    pair_info: dict[str, tuple[str, str]] = {}
    sample_pair_hashes: list[list[str | None]] = []

    for row_index, row in enumerate(rows):
        identifier = row.get("identifier")
        true_smiles = row.get("true_smiles")
        true_mol = mol_from_smiles(true_smiles)
        if true_mol is None:
            raise ValueError(
                f"Invalid ground-truth SMILES at row {row_index}, "
                f"identifier={identifier}: {true_smiles!r}"
            )

        true_fp = morgan_fp(true_mol)
        true_key = inchi_key_2d_from_mol(true_mol)

        raw_predictions = list(row.get("predictions", []))[:10]
        positions: list[dict[str, Any]] = []
        hashes_for_sample: list[str | None] = []

        for rank_index in range(10):
            predicted_smiles = (
                raw_predictions[rank_index]
                if rank_index < len(raw_predictions)
                else None
            )
            predicted_mol = mol_from_smiles(predicted_smiles)

            if predicted_mol is None:
                positions.append(
                    {
                        "rank": rank_index + 1,
                        "smiles": predicted_smiles,
                        "valid": False,
                        "exact": False,
                        "ecfp4": 0.0,
                    }
                )
                hashes_for_sample.append(None)
                continue

            predicted_fp = morgan_fp(predicted_mol)
            predicted_key = inchi_key_2d_from_mol(predicted_mol)
            pair_hash, left, right = symmetric_pair_key(
                true_smiles,
                predicted_smiles,
            )
            pair_info[pair_hash] = (left, right)
            hashes_for_sample.append(pair_hash)

            positions.append(
                {
                    "rank": rank_index + 1,
                    "smiles": predicted_smiles,
                    "valid": True,
                    "exact": predicted_key == true_key,
                    "ecfp4": float(
                        DataStructs.TanimotoSimilarity(
                            true_fp,
                            predicted_fp,
                        )
                    ),
                }
            )

        top1 = positions[0]
        details.append(
            {
                "identifier": identifier,
                "true_smiles": true_smiles,
                "predictions": raw_predictions,
                "positions": positions,
                "valid_predictions": [
                    p["smiles"] for p in positions if p["valid"]
                ],
                "exact1": bool(top1["exact"]),
                "exact10": any(p["exact"] for p in positions),
                "ecfp1": float(top1["ecfp4"]),
                "ecfp10": max(p["ecfp4"] for p in positions),
                "valid1": bool(top1["valid"]),
                "valid_count": sum(p["valid"] for p in positions),
            }
        )
        sample_pair_hashes.append(hashes_for_sample)

    mces_errors: list[dict[str, str]] = []
    computed_mces: dict[str, float] = {}
    cache: MCESCache | None = None

    if args.with_mces:
        solver = choose_solver()
        cache_path = (
            Path(args.mces_cache)
            if args.mces_cache
            else output_path.parent / "mces_cache.sqlite"
        )
        cache = MCESCache(cache_path)

        all_hashes = sorted(pair_info)
        computed_mces.update(
            cache.get_many(
                all_hashes,
                args.mces_threshold,
                myopic_version,
                pulp_version,
            )
        )

        missing_hashes = [
            pair_hash
            for pair_hash in all_hashes
            if pair_hash not in computed_mces
        ]

        print(
            json.dumps(
                {
                    "myopic_mces_version": myopic_version,
                    "pulp_version": pulp_version,
                    "solver": solver,
                    "mces_threshold": args.mces_threshold,
                    "unique_pairs": len(all_hashes),
                    "cache_hits": len(computed_mces),
                    "pairs_to_compute": len(missing_hashes),
                },
                ensure_ascii=False,
                indent=2,
            )
        )

        if missing_hashes:
            tasks = [
                (
                    pair_hash,
                    pair_info[pair_hash][0],
                    pair_info[pair_hash][1],
                    args.mces_threshold,
                    solver,
                )
                for pair_hash in missing_hashes
            ]

            new_cache_rows = []
            with ProcessPoolExecutor(
                max_workers=args.mces_workers
            ) as executor:
                futures = {
                    executor.submit(mces_worker, task): task
                    for task in tasks
                }

                completed = 0
                total = len(futures)
                for future in as_completed(futures):
                    task = futures[future]
                    pair_hash, distance, error = future.result()
                    completed += 1

                    if error is not None:
                        mces_errors.append(
                            {
                                "pair_hash": pair_hash,
                                "smiles_a": task[1],
                                "smiles_b": task[2],
                                "error": error,
                            }
                        )
                    else:
                        assert distance is not None
                        computed_mces[pair_hash] = distance
                        new_cache_rows.append(
                            (
                                pair_hash,
                                task[1],
                                task[2],
                                args.mces_threshold,
                                myopic_version,
                                pulp_version,
                                distance,
                            )
                        )

                    if completed % 25 == 0 or completed == total:
                        print(
                            f"MCES progress: {completed}/{total}",
                            flush=True,
                        )

            cache.put_many(new_cache_rows)

        if mces_errors:
            error_path = output_path.with_suffix(
                output_path.suffix + ".mces_errors.json"
            )
            error_path.write_text(
                json.dumps(
                    {
                        "num_errors": len(mces_errors),
                        "errors": mces_errors,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            cache.close()
            raise RuntimeError(
                f"{len(mces_errors)} MCES pairs failed. "
                f"No metric file was written. Diagnostics: {error_path}"
            )

        for sample_index, detail in enumerate(details):
            distances = []
            for pair_hash in sample_pair_hashes[sample_index]:
                if pair_hash is None:
                    distances.append(INVALID_MCES_PENALTY)
                else:
                    distances.append(computed_mces[pair_hash])

            detail["mces1"] = float(distances[0])
            detail["mces10"] = float(
                min(min(distances), INVALID_MCES_PENALTY)
            )

        cache.close()

    n = max(1, len(details))
    summary: dict[str, Any] = {
        "num_samples": len(details),
        "Top-1 Accuracy": sum(d["exact1"] for d in details) / n,
        "Top-10 Accuracy": sum(d["exact10"] for d in details) / n,
        "ECFP4 Tanimoto@1": sum(d["ecfp1"] for d in details) / n,
        "ECFP4 Tanimoto@10": sum(d["ecfp10"] for d in details) / n,
        "Validity@1": sum(d["valid1"] for d in details) / n,
        "Average valid structures@10": (
            sum(d["valid_count"] for d in details) / n
        ),
    }

    if args.with_mces:
        summary.update(
            {
                "MCES@1": sum(d["mces1"] for d in details) / n,
                "MCES@10": sum(d["mces10"] for d in details) / n,
                "mces_error_pairs": 0,
                "mces_threshold": args.mces_threshold,
                "mces_invalid_penalty": INVALID_MCES_PENALTY,
                "myopic_mces_version": myopic_version,
                "pulp_version": pulp_version,
            }
        )

    output_path.write_text(
        json.dumps(
            {
                "summary": summary,
                "details": details,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        raise
