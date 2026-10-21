# External data, results & model weights

This repository ships **code only**. Every dataset, intermediate result, and model
weight file is hosted externally because they are far too large for a plain git
repository (total ≈ 28 GB; individual files up to ~4.7 GB). Download the items you
need and place them under the paths the scripts default to (overridable via CLI
args / env vars).

**Hosts:** [ModelScope](https://www.modelscope.cn) for model weights;
[Zenodo](https://zenodo.org) for datasets and intermediate results. Model links
are filled in below; Zenodo DOIs are placeholders (`TBD`) — fill them in after
upload.

## Minimal reproduction set

To reproduce the published results end-to-end you need at minimum:

1. `MassSpecGym1.5.json` — the source dataset (N = 17,556 test spectra).
2. The three **Qwen3-14B LoRA adapters** (Exp-1 listwise `checkpoint-1227`,
   Exp-1 Override-V3 `checkpoint-80`, Exp-2 Pairwise Verifier `checkpoint-194`).
3. Exp-1 XGBoost model `preranker_xgb_train20000_val3000.json`.
4. `final_test17556_xgb_llm_v3.jsonl` — Exp-1's final prediction; this is the
   **Exp-1 → Exp-2 bridge** consumed by `145_convert_exp1_to_multiformula.py`.
5. Exp-2 cross-formula XGBoost OOF bundle `xgb_oof_v1/` (OOF scores + predictions).
6. `final_margin01_predictions.jsonl` — Exp-2's final structure predictions.

Everything below is additional (full intermediate-result set for strict auditability).

## Datasets → Zenodo

| Asset | Size | Link | Role |
|---|---|---|---|
| `MassSpecGym1.5.json` | 1.2 GB | `TBD` | Primary MS dataset (231,104 spectra; 17,556 test) |
| `MassSpecGym1.5_retrieval_candidates_formula.json` | 0.37 GB | `TBD` | Exp-1 formula candidate pools |
| `MassSpecGym1.5_retrieval_candidates_mass.json` | 0.45 GB | `TBD` | Exp-1 mass candidate pools |

## Model weights → ModelScope

| Asset | Size | Link | Role |
|---|---|---|---|
| Exp-1 Listwise SFT LoRA (`checkpoint-1227`) | ~392 MB | <https://www.modelscope.cn/models/eibicidi/nc-massspec-exp1-listwise-sft> | Exp-1 stage-1 formula reranker adapter |
| Exp-1 Override-V3 LoRA (`checkpoint-80`) | ~372 MB | <https://www.modelscope.cn/models/eibicidi/nc-massspec-exp1-override-v3> | Exp-1 final KEEP/SWAP verifier adapter |
| Exp-2 Pairwise Verifier LoRA (`checkpoint-194`) | ~913 MB | <https://www.modelscope.cn/models/eibicidi/nc-massspec-exp2-pairwise-verifier> | Exp-2 final pairwise structure verifier adapter |
| Exp-1 XGBoost pre-ranker (`preranker_xgb_train20000_val3000.json`) | ~5 MB | <https://www.modelscope.cn/models/eibicidi/nc-massspec-exp1-xgb-preranker> | Exp-1 broad candidate pre-ranker |

Each ModelScope repo ships the full training checkpoint (adapter weights +
`optimizer.pt` / `scheduler.pt` / `rng_state.pth` for resumable training). For
inference-only use, download just `adapter_config.json`, `adapter_model.safetensors`,
`additional_config.json`, and `args.json` (the LoRA adapter alone is ~130 MB).

Base model: **Qwen3-14B** (ModelScope [`Qwen/Qwen3-14B`](https://www.modelscope.cn/models/Qwen/Qwen3-14B)),
not redistributed here.

## Intermediate results → Zenodo (the large JSONL / binary files)

| Asset | Size | Link | Role |
|---|---|---|---|
| Exp-1 broad pre-ranker features (`preranker_*.X.f32.bin` train/val/test) | ~2.0 GB | `TBD` | Exp-1 XGB feature matrices |
| `preranker_test17556_all_top30_xgb20k_val3000.jsonl` | ~120 MB | `TBD` | Exp-1 XGB Top-30 shortlist |
| `preranker_test17556_gate015_only.jsonl` | ~10 MB | `TBD` | Exp-1 LLM gated subset (z < 0.15) |
| `override_v3_test17556_gate015_pred.jsonl` | small | `TBD` | Exp-1 V3 KEEP/SWAP decisions |
| **`final_test17556_xgb_llm_v3.jsonl`** | 122 MB | `TBD` | **Exp-1 final prediction → Exp-2 input** |
| Exp-1 SFT training data (`formula_nothink_sft_v4_train10k.jsonl`, `override_sft_v3_*_mixed.jsonl`) | ~80 MB | `TBD` | Exp-1 LLM training data |
| Exp-2 `top10_sirius_candidates.jsonl` | ~4.7 GB | `TBD` | Exp-2 SIRIUS/MSNovelist raw candidates |
| Exp-2 `global_formula_structure_pool.jsonl` | ~1.1 GB | `TBD` | Exp-2 global candidate pool |
| Exp-2 `branch_candidates_after_lattice.jsonl` | ~1.0 GB | `TBD` | Exp-2 post-Formula-Lattice candidates |
| Exp-2 pairwise supervised/context JSONL | ~32 MB | `TBD` | Exp-2 Pairwise Verifier training data |
| Exp-2 cross-formula XGBoost OOF bundle (`xgb_oof_v1/`) | ~4.0 GB | `TBD` | Exp-2 OOF scores, features, labels & predictions |
| **`final_margin01_predictions.jsonl`** | ~20 MB | `TBD` | **Exp-2 final structure predictions** |
| Formula-K mechanism dirs (`pred_top1/3/5/10/`, `reference_formula/`) | ~20 GB | `TBD` | Formula-K sensitivity / mechanism experiments |

### Suggested local layout (matches script defaults)

```
nc-massspec-pipeline/
├── data/
│   ├── exp1/          Exp-1 datasets + SFT training data + final prediction
│   └── exp2/          Exp-2 SIRIUS candidates + lattice/pool + pairwise data
├── models/
│   ├── Qwen3-14B/                     (base model, from ModelScope)
│   ├── exp1_listwise/checkpoint-1227/
│   ├── exp1_override_v3/checkpoint-80/
│   └── exp2_pairwise_verifier/checkpoint-194/
└── work/predicted_formula_multihyp_v1/   Exp-2 intermediate `work/` tree
```

Paths are overridable — Exp-1/Exp-2 scripts use `argparse` (`--input`/`--output`)
and the shell scripts use env vars (`MODEL`, `ADAPTERS`, `DATASET`, `PROJECT_ROOT`, …).
