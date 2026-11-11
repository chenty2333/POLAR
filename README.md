# POLAR

**Fractional spectral link recommendation via pole-shared Krylov sketching.**

This repository contains the implementation, experiment scripts, raw results,
and extended paper for POLAR.

Given a graph and a budget of `k` new links, which links should be added?
Degree, log-determinant, and inverse-trace criteria each favor a fixed kind of
improvement, such as reinforcing peripheral vertices or connecting regions.
POLAR instead maximizes a regularized fractional inverse-trace gain whose
exponent `s` makes this preference adjustable:

```math
f_s(S) = \mathrm{Tr}\big((L+\sigma I)^{-s}\big) - \mathrm{Tr}\big((L+L_S+\sigma I)^{-s}\big), \qquad s \in (0,1).
```

Up to a positive scale, `f_s` is the average heat-trace reduction over
diffusion times with mean `s/σ`, so larger `s` weights longer diffusion times.
Greedy optimization is costly: fractional marginals have no single-resolvent
rank-one update, `f_s` is not submodular, and every round needs new linear
solves.

## Method

- **Accurate marginals.** Positive Stieltjes quadrature gives each candidate's
  marginal with a relative error independent of graph size; submodularity-ratio
  and curvature bounds turn this into an error-aware greedy guarantee.
- **Shared solves.** All quadrature poles share one multi-shift Krylov
  iteration, and endpoint sketches and score caches persist across rounds:
  `k` rounds need at most `2q + kB` multi-shift solves instead of `2kq`
  (`q` sketch dimension, `B` shortlist size).
- **Shortlist re-ranking.** Each round re-scores a `B`-edge proxy shortlist
  with algebraic quadrature marginals before selecting.

On eighteen KONECT/SNAP networks (up to 4.0M vertices, 34.7M edges), POLAR
matches exhaustive optima on four small networks and reaches 99.2–100% of
exact fractional greedy on five medium networks. With `B = 1` it is
8.9–17.8× faster than recomputing sketches every round, with identical
selections, and finishes com-LiveJournal in about 15 minutes on one RTX 4090.

## Repository layout

```
paper/          extended version with complete proofs (PDF and LaTeX)
realworld/      GPU implementation and real-network experiments
  polar_gpu.py    POLAR and the Naive, sketched, Degree, Random baselines
  run_real.py     driver: time | effect | opt | param
  results/        raw per-run JSON and logs
synthetic/      CPU scripts: controlled multiscale benchmark, quadrature and
                maintenance checks, sparse scalability
```

## Quick start

GPU experiments need Python 3.12, NumPy, SciPy, pandas, and CuPy 14 on an
NVIDIA GPU (we used one RTX 4090, 24 GB). The `synthetic/` scripts need only
NumPy and SciPy.

```bash
pip install numpy scipy pandas cupy-cuda12x          # choose the CuPy wheel for your CUDA
cd realworld
python fetch.py data/raw $(cat sources.txt)          # download all 18 inputs, check SHA-256
python run_real.py time   --graphs com-DBLP --s 0.5  # running time vs. per-round recomputation
python run_real.py effect --graphs ca-GrQc --slq     # selection quality vs. exact greedy and baselines
python report.py                                     # rebuild REPORT_realworld.md from results/
```

[`realworld/README.md`](realworld/README.md) lists every script, dataset
(with SHA-256), and result file;
[`synthetic/effectiveness/README.md`](synthetic/effectiveness/README.md)
describes the multiscale benchmark.

## Companion project

[FastFrac](https://github.com/chenty2333/FastFrac) studies the positive-power
trace `Tr((I + αL)^s)`, which is submodular and admits a different marginal
reduction.
