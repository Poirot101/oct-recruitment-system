# RWTH materials research

Research workspace for Dr Soumya's (RWTH) project on cutting the compute time of
molecular-mechanics simulations of disordered solids.

- **[analysis/ANALYSIS.md](analysis/ANALYSIS.md)**: in-depth analysis of the two papers, the
  prototype's results and the recommended changes. Start here.
- `materials/`: the reference papers (see `materials/README.md`).
- `src/`: the prototype code (2D glass simulator, accelerated local yield stress scan,
  soft-mode reduced-order model, GNN surrogates) and the scripts behind every number
  (see `src/README.md`).
- `results/`: the generated glasses, yield-stress maps, trained models and metrics.

Headline: adaptive certified strain stepping with a looser minimiser tolerance gives the exact
local yield stress map 18× faster, with every yield stress still matching the baseline. A
soft-mode reduced-order model ranks weak sites about 75× faster still (Spearman ≈ 0.5 on the
weakest direction), but screening with it saves only about 2× more at 89% recall. Across 200
real plastic events, the exact map is the best predictor. A review of 48 papers is in
[analysis/LITERATURE.md](analysis/LITERATURE.md).
