# MassSpecGym 1.5 — Molecular-Formula Prediction & De-Novo Structure Generation

Code for the Nature Communications manuscript on small-molecule identification from
tandem mass spectrometry, evaluated on the **MassSpecGym 1.5** benchmark
(test split, N = 17,556 spectra).

This repository contains **code only**. Datasets, intermediate results, and model
weights are hosted externally — see [`DATA.md`](DATA.md).

## What the pipeline does

Two sequential experiments form one pipeline:

```
                        Experiment 1                              Experiment 2
   MS/MS spectra  ──────────────────────────►   top-K formula    ──────────────────────────────►  top-K structures
   (MassSpecGym)   XGBoost broad pre-ranker      hypotheses        SIRIUS/MSNovelist de-novo
                   → Qwen3-14B Override-V3                         Formula-Lattice (cross-spectrum transfer
                   verifier (KEEP/SWAP Top-1)                       + graph-edit expansion)
                                                                    cross-formula XGBoost OOF ranking
                                                                    Qwen3-14B Pairwise Verifier (Rank1 vs 2–5)
```

Experiment 1's output (`final_test17556_xgb_llm_v3.jsonl`) is the **upstream input**
to Experiment 2 (via `experiment2_structure_generation/code/145_convert_exp1_to_multiformula.py`).

## Final results (test split, N = 17,556)

### Experiment 1 — molecular-formula prediction

| Metric | Value |
|---|---|
| Top-1 Accuracy | 63.87 % |
| Top-3 Accuracy | 77.34 % |
| Top-5 Accuracy | 81.44 % |
| Mass-valid Rate (|ppm| ≤ 5) | 91.25 % |
| Formula Validity | 100.00 % |
| Adduct Accuracy | 79.90 % |

### Experiment 2 — de-novo structure generation

| Metric | Value |
|---|---|
| Top-1 Accuracy (2D InChIKey exact) | 22.75 % |
| Top-10 Accuracy | 29.22 % |
| ECFP4@1 | 0.4453 |
| ECFP4@10 | 0.5311 |
| MCES@1 | 14.7363 |
| MCES@10 | 10.6072 |
| Empty outputs (after Formula-Lattice) | 13 |

The formal main model uses **Predicted Formula Top-10** and does **not** include
terminal empty-pool rescue. See the per-experiment READMEs for the full definitions.

## Repository layout

```
nc-massspec-pipeline/
├── README.md                              (this file)
├── DATA.md                                external-asset table (HuggingFace / Zenodo)
├── requirements.txt
├── docs/                                  per-file manifests (verbatim, from the authors)
├── experiment1_formula_prediction/code/   10 .py + 3 .sh
├── experiment2_structure_generation/code/  31 .py + 3 .sh  (+ multiformula_common.py — see below)
├── figures/                               11 self-contained matplotlib scripts + reference renders
```

Script filenames keep their numeric prefixes (`131_`, `132_` … `248_`) because those
numbers **encode the pipeline order** — run them in ascending order within each
experiment. Detailed per-file descriptions are in
[`docs/experiment1_file_manifest.txt`](docs/experiment1_file_manifest.txt) and
[`docs/experiment2_file_manifest.txt`](docs/experiment2_file_manifest.txt).

## Environment

Python dependencies are pinned in [`requirements.txt`](requirements.txt). Key
third-party packages: **RDKit** (2026.03.4 per the paper), **XGBoost**,
**pulp==2.9.0** + **myopic-mces==1.0.0** (for MassSpecGym-compatible MCES),
**pandas**, **scikit-learn**, **torch**.

External (non-pip) tools, also required for a full re-run:
- **SIRIUS 6.3.7** + Java — formula-conditioned de-novo structure generation
  (`143_run_multiformula_sirius_sharded.sh`).
- **ms-swift** + **vLLM** — Qwen3-14B LoRA SFT and deploy
  (`train_*.sh`, `deploy_override_v3_FINAL.sh`).

## How to reproduce

End-to-end reproduction requires the hosted data/models in [`DATA.md`](DATA.md).
The high-level steps:

1. Download the **minimal reproduction set** listed at the top of `DATA.md` and
   place it under `data/` and `models/` (paths the scripts default to).
2. **Experiment 1** — see [`experiment1_formula_prediction/README.md`](experiment1_formula_prediction/README.md).
3. **Experiment 2** — see [`experiment2_structure_generation/README.md`](experiment2_structure_generation/README.md).
   Experiment 2 consumes Experiment 1's final prediction as its formula-hypothesis input.

The figure scripts in [`figures/`](figures/) are self-contained and need no hosted
data — run them directly.

## ⚠️ Known gap: `multiformula_common.py`

Six Experiment-2 scripts (`131`, `132`, `133`, `134`, `136`, `140`) import a local
helper module, `multiformula_common`, that is **not in this repository** (it was
absent from the source backup). It must be retrieved and dropped into
`experiment2_structure_generation/code/`. See
[`experiment2_structure_generation/code/MISSING-multiformula_common.md`](experiment2_structure_generation/code/MISSING-multiformula_common.md).
Until then, those six scripts cannot run; the rest of the repository is unaffected.

## License

_(TBD — add the project's chosen license before public release.)_
