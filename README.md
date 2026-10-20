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

Headline: adaptive certified strain stepping gives the exact local yield stress map 8–13×
faster. A soft-mode reduced-order model with a learned corrector ranks weak sites
about 100× faster still (Spearman ≈ 0.45), which is enough to screen the weakest sites
before running the exact scan.
