# POLAR: controlled degree-heterogeneity experiment

Run `python 3 polar_effectiveness_revision.py` from any directory. It writes
`results.json` and `results.csv` beside the script. Dependencies: NumPy and
SciPy. The script uses one BLAS thread and pins to the second allowed CPU
(or the only allowed CPU); this execution used CPU 1.

## Fixed experiment

The control is exactly the previous n=800, seed 711 benchmark: two 390-vertex
cores, each the union of three independent Hamilton cycles of weight 10,
a weight 0.5 intercore bridge, and 20 weight 0.3 leaves. The revised stress
benchmark adds five vertices forming a C5 with weight 0.5 edges, each
independently attached by a weight 0.5 edge to core vertices 20,...,24.
Thus n=805 and the active candidate endpoint count is 125. Graph seeds
711,712,713 were fixed before the three-seed evaluation; all are retained.

Every algorithm receives the same 3000 candidates: uniformly sample 2995
of the original eligible intercore/leaf-to-core links, add all five missing
C5 chords, and shuffle with the graph RNG. The original control preserves
its original 3000-candidate sampling rule. Budgets k=15; sigma=.01;
POLAR p=20,q=24,B=4; CG relative residual tolerance 1e-7. Sketch RNG seed 2026
is shared across algorithms. Node sketches use centered global orthogonal
columns; incidence sketches use global orthogonal columns.

Degree means minimum current **weighted** endpoint-degree sum. Each
C5 vertex begins at degree 1.5. At any time an unselected chord has endpoint
sum<=5, since each endpoint can have received at most its one other missing
chord. Leaf-to-core candidates have sum>=60.3 and intercore candidates
sum>=120. Therefore Degree deterministically spends its first five
selections on local chords, then chooses ten different leaves. This is a
controlled stress benchmark for local degree ranking, not a claimed real
network or a universal advantage over Degree. The old control, where Degree
is competitive, remains in the output.

## Exact fractional reference

The reference uses the paper's positive composite Gauss-Jacobi/Legendre
quadrature and the proved derivative relative-error bound 256*4^(-r).
It selects r=19 and 304 poles on these spectra, giving a bound
9.313225746154785e-10. The candidate screen uses an outward guard 1e-9;
every interval-surviving contender is evaluated by `numpy.linalg.eigvalsh`.
Thus an exhaustive true-marginal greedy reference is obtained through
certified analytic quadrature screening and ordinary double-precision
spectral evaluation. At most two contenders survived in any recorded
round. Raw data records every round's pole count, bound, true marginal,
quadrature score, selected candidate IDs and selected endpoint pairs.

## Summary

All gains are evaluated from the initial/final eigenvalues. Reported
standard deviations use ddof=1 over the three stress graph seeds.
Percent improvements are ratios of mean gains, not maximum gains.

| exponent | POLAR over Degree | over Exact-ER | over Sketched-ER | Exact-greedy recovery |
|---|---:|---:|---:|---:|
|0.25|26.8124%|48.0579%|19.3078%|99.9519%|
|0.5|36.9396%|21.7660%|5.6537%|99.9192%|

The 5.65% improvement over Sketched-ER at s=.5 should be reported as such;
it is not a 20–40% improvement.

`exploration.json` contains both pilot weight configurations at seed 711
(cycle/anchor weights 0.5/0.5 and 1/1). The 0.5/0.5 construction was locked
before the 3-seed evaluation. Pilots are not pooled into the headline
statistics. `summary.json` combines the final aggregate data, graph sizes,
and separate pilot record. Historical elapsed times are not mixed with
these measurements; this experiment's purpose is effectiveness, and its
single-run solver times are auxiliary.
