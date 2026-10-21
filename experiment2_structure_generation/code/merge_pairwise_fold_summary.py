import argparse
import json
from pathlib import Path


ap = argparse.ArgumentParser(
    description="Summarize pairwise-verifier five-fold OOF metrics"
)
ap.add_argument(
    "--root",
    default="work/predicted_formula_multihyp_v1/cross_formula/xgb_oof_v1/pairwise_verifier_v2",
    help="pairwise_verifier_v2 root directory",
)
args = ap.parse_args()

ROOT = Path(args.root)


folds = {
    1: ROOT / "fold1_pilot/fold1_pairwise_metrics.json",
    2: ROOT / "fold2_oof/fold2_pairwise_metrics.json",
    3: ROOT / "fold3_oof/fold3_pairwise_metrics.json",
    4: ROOT / "fold4_oof/fold4_pairwise_metrics.json",
    5: ROOT / "fold5_oof/fold5_pairwise_metrics.json",
}


total = {
    "samples": 0,
    "baseline_hits": 0,
    "llm_hits": 0,
    "fix": 0,
    "break": 0,
}


print("=" * 80)
print("PAIRWISE VERIFIER FIVE-FOLD SUMMARY")
print("=" * 80)


for fold, path in folds.items():

    x = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    rows = x["policies"]["final_lowest_rank_confirmed"]["gate_sweep"]

    row = [
        r for r in rows
        if abs(r["threshold"] - 0.5) < 1e-8
    ][0]


    n = x["num_nonempty_samples"]

    baseline = x["baseline_xgb_top1_hits"]

    llm = row["top1_hits"]

    fix = row["fixed_errors"]

    brk = row["broken_correct"]


    print(
        f"Fold {fold}: "
        f"N={n}, "
        f"XGB={baseline}, "
        f"LLM={llm}, "
        f"Fix={fix}, "
        f"Break={brk}, "
        f"Delta={llm-baseline}"
    )


    total["samples"] += n
    total["baseline_hits"] += baseline
    total["llm_hits"] += llm
    total["fix"] += fix
    total["break"] += brk


print()
print("=" * 80)
print("TOTAL")
print("=" * 80)


print(
    "Samples:",
    total["samples"]
)

print(
    "XGB:",
    total["baseline_hits"],
    "/",
    total["samples"],
    "=",
    f"{100*total['baseline_hits']/total['samples']:.2f}%"
)

print(
    "Pairwise:",
    total["llm_hits"],
    "/",
    total["samples"],
    "=",
    f"{100*total['llm_hits']/total['samples']:.2f}%"
)

print(
    "Fix:",
    total["fix"]
)

print(
    "Break:",
    total["break"]
)

print(
    "Delta:",
    total["llm_hits"]-total["baseline_hits"]
)

print(
    "Delta pp:",
    f"{100*(total['llm_hits']-total['baseline_hits'])/total['samples']:.2f}"
)

print("=" * 80)
