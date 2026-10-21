# Experiment 2 — De-Novo Structure Generation

Generate molecular structures de novo from tandem-MS spectra under
molecular-formula uncertainty. The top-K (K=10) formula hypotheses from
Experiment 1 are each expanded into independent SIRIUS/MSNovelist
structure-generation branches; a **Multi-Hypothesis Formula-Lattice** transfers
candidates across spectra and formula neighbours; a cross-formula XGBoost ranks
the merged global pool (identity-disjoint 5-fold OOF); and a Qwen3-14B
**Pairwise Verifier** selectively re-ranks Rank-1 vs Rank-2–5 on hard cases.

```
Exp-1 final formula predictions
  → Predicted Formula Top-10  →  independent formula branches
  → SIRIUS 6.3.7 / MSNovelist de-novo generation  (143_run_multiformula_sirius_sharded.sh)
  → strict branch-level chemical validation
  → Multi-Hypothesis Formula-Lattice            (140 + 144_run_global_multihyp_lattice.sh)
  → Global Formula-Structure Pool               (136)
  → Cross-formula XGBoost OOF ranking           (150 → 210 → 151)
  → Qwen3-14B Pairwise Verifier (margin 0.1)    (160 → 162 → 167/171)
  → Final Top-10 structures  →  Accuracy / ECFP4 / MCES  (168 / compute_mces / 04_evaluate)
```

## Final results (test, N = 17,556)

Top-1 22.75 % · Top-10 29.22 % · ECFP4@1 0.4453 · ECFP4@10 0.5311 ·
MCES@1 14.7363 · MCES@10 10.6072 · Empty (after Formula-Lattice) 13.

Final LLM: **Pairwise Verifier / checkpoint-194** (4,652 SFT records, 1 epoch).
Main model uses **Predicted Formula Top-10** and does **not** include terminal
empty-pool rescue.

## Scripts (run in this order; numeric prefixes encode the order)

| Stage | Script | Notes |
|---|---|---|
| **Input / branches** | | |
| convert Exp-1 output | `145_convert_exp1_to_multiformula.py` | **Exp-1 → Exp-2 bridge**; reads `final_test17556_xgb_llm_v3.jsonl` |
| validate input | `131_validate_multiformula_input.py` | schema check |
| leakage audit | `132_check_no_gt.py` | proves no GT in predictor input |
| build branches | `133_build_multiformula_branches.py` | one branch per formula hypothesis |
| SIRIUS input | `134_build_multiformula_sirius_input.py` | per-branch `.ms` input |
| **SIRIUS / MSNovelist** | | |
| run sharded SIRIUS | `143_run_multiformula_sirius_sharded.sh` | **Env:** `PROJECT_ROOT`, `SIRIUS_BIN`, `BRANCH_MANIFEST`, `SIRIUS_INPUT_ROOT`, `RUN_ROOT`, `MERGED_CANDIDATES` (all required) |
| parse denovo summary | `138_parse_multiformula_denovo_summary.py` | parses `denovo_structure_identifications_top-100.tsv` |
| (legacy SIRIUS helpers) | `82_prepare_sirius_input.py`, `83_parse_denovo_summary.py` | earlier single-formula versions |
| **Formula-Lattice** | | |
| core chemistry | `80_formula_lattice_predict.py` | graph-edit operators (dehydration, deacetylation, hydroxylation, reduction) |
| multi-hyp predictor | `140_multihyp_formula_lattice_predict.py` | same-formula transfer + formula-neighbour graph + Top-1 lock |
| run lattice | `144_run_global_multihyp_lattice.sh` | **Env:** `PROJECT_ROOT`, `BRANCH_MANIFEST`, `MERGED_CANDIDATES`, `LATTICE_ROOT` |
| global pool | `136_build_global_formula_structure_pool.py` | canonical-SMILES dedup, provenance merge |
| **Cross-formula XGBoost** | | |
| rank dataset | `150_build_cross_formula_rank_dataset.py` | label = 2D InChIKey first-block match |
| add RDKit features | `210_add_rdkit_features.py` | `--input`/`--output` (defaults under `work/…`) |
| OOF ranker | `151_oof_xgb_cross_formula_ranker_rdkit.py` | GroupKFold 5-fold, identity-disjoint |
| eval OOF | `152_eval_oof_cross_formula.py`, `146_evaluate_pre_cross_formula_pool.py` | |
| **Pairwise Verifier** | | |
| build SFT data | `160_build_pairwise_verifier_v2.py` | A/B pairs, balanced labels (no GT in prompt) |
| prepare fold-1 | `161_prepare_pairwise_fold1.py` | |
| OOF folds 2–5 | `164_prepare_pairwise_folds_2to5.sh` | **Env:** `PROJECT_ROOT` (defaults to repo root) |
| run pairwise inference | `162_run_pairwise_fold1_api.py` | OpenAI-compatible chat API; **also used for final test**; challenger-first, double-confirmation; `final_lowest_rank_confirmed` policy; margin=0.1 |
| merge OOF | `167_merge_pairwise_oof.py` | |
| fix merged preds | `171_fix_merge_pairwise_predictions.py` | re-syncs candidate_ids/predictions after promotion; `--base` |
| eval pairwise | `168_eval_pairwise_oof_fast.py`, `163_eval_pairwise_fold1.py`, `merge_pairwise_fold_summary.py` | |
| **Evaluation** | | |
| fast metrics | `168_eval_pairwise_oof_fast.py` | Top-1/10 exact (2D InChIKey), ECFP4 |
| MCES | `compute_mces.py`, `04_evaluate.py` | myopic-mces==1.0.0, pulp==2.9.0, threshold=15, invalid→100 |
| attach truth (eval only) | `148_attach_truth_for_denovo_eval.py` | |
| **Mechanism experiments** | | |
| oracle/reference formula | `246_build_oracle_single_formula_input.py`, `oracle_formula_chemistry.py` | upper-bound with reference formula (GT adduct still not input) |
| top-10 input | `248_build_predicted_top10_input.py` | builds the Top-10 main-model input |

## ⚠️ Required: `multiformula_common.py`

Six scripts here (`131`, `132`, `133`, `134`, `136`, `140`) `import
multiformula_common`, which is **not in this repo**. Retrieve it per
[`MISSING-multiformula_common.md`](code/MISSING-multiformula_common.md) and drop
it next to these scripts before running the pipeline.

## Evaluation rules (must match code)

- **Exact structure:** 2D InChIKey first-block exact match.
- **ECFP4:** RDKit Morgan, radius=2, nBits=2048; Top-K = max Tanimoto over first K; missing/invalid = 0.
- **MCES:** myopic-mces==1.0.0, PuLP==2.9.0, threshold=15, `always_stronger_bound=True`,
  invalid/missing = 100, MCES@K = min distance over first K, cap=100; all 17,556
  samples remain in the denominator.

## Formula-K sensitivity (mechanism)

`pred_top1 / pred_top3 / pred_top5 / pred_top10 / reference_formula` sweep the
number of retained formula hypotheses. Top-1 16.50 → 21.13 → 22.37 → 22.75 %
as K grows 1→3→5→10; reference-formula upper bound = 26.21 %. Formula-Lattice
empty-pool recovery rises from 89.7 % (K=1) to 99.4 % (K=10). See
`docs/experiment2_file_manifest.txt` for the full table.
