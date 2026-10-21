# Experiment 1 — Molecular-Formula Prediction

Predict the neutral molecular formula of each tandem-MS spectrum. A broad
candidate pool is generated from chemical rules, ranked by an XGBoost
learning-to-rank pre-ranker, then a Qwen3-14B **Override-V3** LLM verifier
selectively re-ranks the Top-1 candidate on low-confidence (z-margin < 0.15) cases.

```
MassSpecGym spectra
  → Broad Candidate Pool   (V3 Top50 ∪ OLD Top400 ∪ Direct Top5 ∪ V3-F Top15, dedup)
  → XGBoost global ranking  →  Top-30
  → normalized-margin gate z < 0.15  →  Qwen3-14B Override-V3 (checkpoint-80)
  → KEEP / SWAP merged back into the full test set
  → eval_override_v2.py  →  six official metrics
```

## Final results (test, N = 17,556)

Top-1 63.87 % · Top-3 77.34 % · Top-5 81.44 % · Mass-valid 91.25 % ·
Formula Validity 100.00 % · Adduct 79.90 %.

Final LLM: **Override-V3 / checkpoint-80**. Final XGB: `preranker_xgb_train20000_val3000.json`. Final evaluator: `eval_override_v2.py`.

## Scripts (run in this order)

| Stage | Script | Notes |
|---|---|---|
| Broad pool + features | `build_broad_preranker_dataset.py` | 61-dim candidate features; depends on `eval_final_candidate_union_2000.py`, `mz_formula_candidates.py`, `build_formula_nothink_sft_v4_listwise_zh_resume.py` |
| Train XGB | `train_xgb_broad_preranker.py` | `rank:pairwise`, `ndcg@30` |
| Export Top-30 | `export_preranker_shortlist.py` | XGB inference → `preranker_test17556_all_top30_*.jsonl` |
| Stage-1 SFT data | `build_formula_nothink_sft_v4_listwise_zh_resume.py` | listwise formula reranking data |
| Train stage-1 LLM | `train_formula_nothink_14b_10k.sh` | Qwen3-14B no-thinking listwise LoRA → `checkpoint-1227`. **Env:** `MODEL` (required), `DATASET`, `OUTPUT_DIR` |
| Override-V3 SFT data | `build_override_sft_v3.py` | KEEP/SWAP pairs (SWAP = Rank1≠gold ∧ Rank2=gold) |
| Train Override-V3 | `train_formula_override_14b_v3.sh` | continues from `checkpoint-1227` → `checkpoint-80`. **Env:** `MODEL`, `INIT_ADAPTER` (required), `DATASET`, `VAL_DATASET`, `OUTPUT_DIR` |
| Deploy LLM | `deploy_override_v3_FINAL.sh` | ms-swift + vLLM, temperature=0, max_new_tokens=32, thinking off. **Env:** `MODEL`, `ADAPTERS` (required) |
| LLM inference | `infer_override_deploy.py` | calls the deployed endpoint; KEEP/SWAP on the z<0.15 gated subset |
| Merge | `merge_override_full_test.py` | gated decisions merged back into the full 17,556 |
| Evaluate | `eval_override_v2.py` | the official six-metric evaluator |

All scripts drive I/O through `argparse` (`--input`/`--output`); the shell scripts
use env vars. Run `python <script> --help` for the full arg list.

## Metric definitions

- **Top-1/3/5 Accuracy** — true neutral formula in the final Top-1/3/5.
- **Mass-valid Rate** — Top-1 candidate within 5 ppm.
- **Formula Validity** — Top-1 formula parses, DBE-valid, nitrogen-rule-valid,
  element-ratio-valid. (≠ formula accuracy.)
- **Adduct Accuracy** — Top-1 latent BestAdduct matches the true adduct
  (true adduct used for evaluation only, not as an input).

## Data isolation

- XGB trained on the MassSpecGym **train** split (`preranker_train20000`, 18,067
  ranking groups / 2,488,266 candidate rows).
- XGB validation: `preranker_val3000_confirm` (2,643 groups).
- Override-V3 hard-mining **excludes** XGB-train IDs and the Fresh-HardDev IDs
  (`preranker_train20000_ids_for_exclude.jsonl`, `preranker_fresh_harddev_z030_*`).
- The test split (17,556) is used for final evaluation only; test labels did not
  update `checkpoint-80`'s gradients. See `docs/experiment1_file_manifest.txt`.
