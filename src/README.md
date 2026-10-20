# Code: `lysgnn`

Python + numba prototype for the analysis in `analysis/ANALYSIS.md`.

- `lysgnn/sim.py`: numba kernels. 2D binary LJ glass, FIRE, MD quench, the local yield stress scan
  (baseline and adaptive), and the soft-mode yield estimate.
- `lysgnn/glass.py`: drivers for glass generation, global AQS and LYS maps.
- `lysgnn/gnn.py`: edge-conditioned GNN (pure PyTorch).

## Reproduce (from `src/`, about 1.5 h on 4 cores)

```bash
pip install numpy scipy numba torch scikit-learn matplotlib
export PYTHONPATH=.
python scripts/benchmark_lys.py ../results/bench_glass.npz 48 ../results/benchmark_lys.json
python scripts/make_dataset.py ../results/data 16 17 --aqs
python scripts/make_dataset.py ../results/data 10 11 12 13 14 15
python scripts/softmode_features.py ../results/data ../results/softmode 10 11 12 13 14 15 16 17
python scripts/train_gnn.py ../results/data ../results/gnn.json --train 10 11 12 13 14 --val 15 --test 16 17
python scripts/train_gnn.py ../results/data ../results/gnn_hybrid.json --train 10 11 12 13 14 --val 15 --test 16 17 \
    --epochs 300 --layers 4 --softdir ../results/softmode
python scripts/hybrid_model.py ../results/data ../results/softmode ../results/hybrid.json --train 10 11 12 13 14 15 --test 16 17
cd .. && PYTHONPATH=src python src/scripts/make_figures.py
```

`results/bench_glass.npz` is the benchmark glass (seed 3); `glass.make_glass(seed=3)` regenerates it.

## Round 2 (about 1.5 h more on 4 cores)

```bash
python scripts/benchmark_lys_more.py   # larger coarse steps and looser minimiser tolerance
python scripts/make_dataset.py ../results/data 18 19 20 21 22 23 24 25 26 27 28 29 --aqs
python scripts/add_aqs.py ../results/data 10 11 12 13 14 15
python scripts/softmode_features.py ../results/data ../results/softmode 18 19 20 21 22 23 24 25 26 27 28 29
python scripts/round2_study.py ../results/data ../results/softmode ../results/round2.json
python scripts/round2_figures.py ../results/round2.json ../analysis/figures
```

## Round 3 (about 40 min on 4 cores)

```bash
python scripts/round3_events.py ../results/data ../results/softmode ../results/round2 ../results/round3_events.json --workers 4
python scripts/round3_figures.py ../results/round3_events.json ../analysis/figures
```
