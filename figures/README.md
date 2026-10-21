# Figures

Regeneration scripts for the paper's **Figure 2** (Experiment 1, molecular-formula
identification) and **Figure 3** (Experiment 2, de-novo structure generation).

Each `figX*.py` is **self-contained**: the plotted numbers are inlined as data
literals inside the script, and the only output is written to `./figures/`
(relative to your current working directory). There are **no external data
dependencies** — you do not need any of the hosted data/model artifacts to
regenerate these figures.

## Regenerate a panel

```bash
# from the repository root:
python figures/fig2a.py      # writes ./figures/fig2a.{pdf,svg,png}
python figures/fig3a.py
```

## Layout

| File | Panel |
|---|---|
| `fig2a.py`–`fig2e.py` | Figure 2 a–e — Experiment 1 (formula identification) |
| `fig3a.py`–`fig3f.py` | Figure 3 a–f — Experiment 2 (structure generation) |
| `outputs/` | Reference renders (PDF/PNG/SVG) of each panel, plus the composed `new-fig2.*` / `new-fig3.*`. |

The numbers plotted correspond to the final results reported in the paper and in
the top-level [`README.md`](../README.md). The scripts were renamed from the
original `Fig. 2a.py` etc. only to remove spaces/dots; their content is unchanged.
