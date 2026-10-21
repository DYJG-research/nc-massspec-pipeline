import argparse
import json
from pathlib import Path


ap = argparse.ArgumentParser(
    description="Fix merged pairwise predictions: re-sync candidate_ids/predictions after promotion"
)
ap.add_argument(
    "--base",
    default="work/predicted_formula_multihyp_v1",
    help="work root (predicted_formula_multihyp_v1)",
)
args = ap.parse_args()

base = Path(args.base)

xgb_path = (
    base
    / "cross_formula/xgb_oof_v1/oof_predictions.jsonl"
)

pair_path = (
    base
    / "cross_formula/xgb_oof_v1/"
    "pairwise_verifier_v2/test17556_final_v2/"
    "pairwise_predictions_full.jsonl"
)

out_path = (
    base
    / "cross_formula/xgb_oof_v1/"
    "pairwise_verifier_v2/test17556_final_v2/"
    "final_pairwise_v2_predictions_fixed.jsonl"
)


pair = {}

with pair_path.open(encoding="utf-8") as f:
    for line in f:
        if line.strip():
            x=json.loads(line)
            pair[str(x["sample_id"])] = x


changed = 0
total = 0


with xgb_path.open(encoding="utf-8") as fin, \
     out_path.open("w", encoding="utf-8") as fout:

    for line in fin:

        if not line.strip():
            continue

        row=json.loads(line)

        sid=str(
            row.get("identifier")
            or row.get("sample_id")
        )

        p=pair.get(sid)

        if p is not None:

            target = (
                p["final_lowest_rank_confirmed"]
            )

            old = (
                p["baseline_candidate_id"]
            )

            if target != old:

                ids = list(
                    row["candidate_ids"]
                )

                preds = list(
                    row["predictions"]
                )

                if target in ids:

                    idx = ids.index(target)

                    ids.insert(
                        0,
                        ids.pop(idx)
                    )

                    preds.insert(
                        0,
                        preds.pop(idx)
                    )

                    row["candidate_ids"] = ids
                    row["predictions"] = preds

                    changed += 1

        fout.write(
            json.dumps(
                row,
                ensure_ascii=False
            )
            + "\n"
        )

        total += 1


print("="*80)
print("FINISHED")
print("="*80)
print("Total:", total)
print("Changed:", changed)
print("Output:", out_path)
