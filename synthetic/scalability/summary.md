# POLAR scalability extension

## Protocol

{"cpu_affinity": 0, "python": "3.12.14", "numpy": "2.3.5", "scipy": "1.17.0", "cpu_model": "AMD EPYC 9V74 80-Core Processor"}

Parameters: {"p": 20, "q": 24, "k": 15, "N": 3000, "sigma": 1.0, "s": 0.5, "tol": 1e-07, "repetitions": 3}

All times are medians of three runs. Methods run in rotated order. Graph and projection construction are excluded; initial solves, scoring, sketch updates and sparse graph modifications are included. Each mode uses identical input graph, candidates and fixed Gaussian sketches. All added links have weight 1. sigma=1, s=0.5. B=1; only 14 selected-edge solves are needed after 48 initialization right-hand sides. The tight triangular numerator / reverse-triangular denominator envelope uses a rebuilt heap; this is not the new persistent affine heap backend.

Ring-random uses the original ring + 2n uniform random pair recipe, seed n+7. Small-world uses a 6-neighbor ring with rewiring probability 0.1. Preferential uses K4 followed by 3 preferentially sampled neighbors per arriving vertex. These are all synthetic graphs.

| Graph | n | m | nnz(M0) | n_C | Method | SpMVs | Norm evals | Pruned % | Median s | SpMV factor | Time factor | Match |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| ring-random / local | 400 | 1190 | 2780 | 96 | Naive-ShiftedCG | 16893 | 44895 | 0.00 | 1.1821 | 1.00 | 1.00 | 15/15 |
| ring-random / local | 400 | 1190 | 2780 | 96 | Inc-Full | 1437 | 44895 | 0.00 | 0.3748 | 11.76 | 3.15 | 15/15 |
| ring-random / local | 400 | 1190 | 2780 | 96 | POLAR-Lazy | 1437 | 4185 | 90.68 | 0.1802 | 11.76 | 6.56 | 15/15 |
| ring-random / local | 1500 | 4492 | 10484 | 96 | Naive-ShiftedCG | 18300 | 44895 | 0.00 | 1.6495 | 1.00 | 1.00 | 15/15 |
| ring-random / local | 1500 | 4492 | 10484 | 96 | Inc-Full | 1545 | 44895 | 0.00 | 0.4395 | 11.84 | 3.75 | 15/15 |
| ring-random / local | 1500 | 4492 | 10484 | 96 | POLAR-Lazy | 1545 | 4172 | 90.71 | 0.2760 | 11.84 | 5.98 | 15/15 |
| ring-random / local | 4000 | 11988 | 27976 | 96 | Naive-ShiftedCG | 17797 | 44895 | 0.00 | 1.7062 | 1.00 | 1.00 | 15/15 |
| ring-random / local | 4000 | 11988 | 27976 | 96 | Inc-Full | 1500 | 44895 | 0.00 | 0.4506 | 11.86 | 3.79 | 15/15 |
| ring-random / local | 4000 | 11988 | 27976 | 96 | POLAR-Lazy | 1500 | 4207 | 90.63 | 0.2496 | 11.86 | 6.84 | 15/15 |
| ring-random / local | 10000 | 29995 | 69990 | 96 | Naive-ShiftedCG | 18353 | 44895 | 0.00 | 2.9178 | 1.00 | 1.00 | 15/15 |
| ring-random / local | 10000 | 29995 | 69990 | 96 | Inc-Full | 1546 | 44895 | 0.00 | 0.5583 | 11.87 | 5.23 | 15/15 |
| ring-random / local | 10000 | 29995 | 69990 | 96 | POLAR-Lazy | 1546 | 4273 | 90.48 | 0.3344 | 11.87 | 8.73 | 15/15 |
| ring-random / local | 20000 | 59985 | 139970 | 96 | Naive-ShiftedCG | 19080 | 44895 | 0.00 | 6.2155 | 1.00 | 1.00 | 15/15 |
| ring-random / local | 20000 | 59985 | 139970 | 96 | Inc-Full | 1598 | 44895 | 0.00 | 0.7933 | 11.94 | 7.84 | 15/15 |
| ring-random / local | 20000 | 59985 | 139970 | 96 | POLAR-Lazy | 1598 | 4177 | 90.70 | 0.6078 | 11.94 | 10.23 | 15/15 |
| small-world / local | 10000 | 30000 | 70000 | 96 | Naive-ShiftedCG | 18652 | 44895 | 0.00 | 2.8805 | 1.00 | 1.00 | 15/15 |
| small-world / local | 10000 | 30000 | 70000 | 96 | Inc-Full | 1593 | 44895 | 0.00 | 0.5663 | 11.71 | 5.09 | 15/15 |
| small-world / local | 10000 | 30000 | 70000 | 96 | POLAR-Lazy | 1593 | 5033 | 88.79 | 0.3487 | 11.71 | 8.26 | 15/15 |
| preferential / local | 10000 | 29994 | 69988 | 96 | Naive-ShiftedCG | 51395 | 44895 | 0.00 | 6.7032 | 1.00 | 1.00 | 15/15 |
| preferential / local | 10000 | 29994 | 69988 | 96 | Inc-Full | 4301 | 44895 | 0.00 | 0.7620 | 11.95 | 8.80 | 15/15 |
| preferential / local | 10000 | 29994 | 69988 | 96 | POLAR-Lazy | 4301 | 3700 | 91.76 | 0.5570 | 11.95 | 12.03 | 15/15 |
| ring-random / global | 10000 | 29995 | 69990 | 4544 | Naive-ShiftedCG | 18360 | 44895 | 0.00 | 6.6768 | 1.00 | 1.00 | 15/15 |
| ring-random / global | 10000 | 29995 | 69990 | 4544 | Inc-Full | 1547 | 44895 | 0.00 | 1.1757 | 11.87 | 5.68 | 15/15 |
| ring-random / global | 10000 | 29995 | 69990 | 4544 | POLAR-Lazy | 1547 | 3026 | 93.26 | 0.7995 | 11.87 | 8.35 | 15/15 |

The n=10000 global-support diagnostic changes n_C from 96 to 4544. It retains the same N=3000 candidate count but samples endpoints uniformly over all graph vertices; this is a different candidate-selection workload, not an isolated n_C-only experiment.

Every run and every repeat matched the corresponding recomputed fixed-projection sequence exactly (15/15). The observed small-world pruning rate is 88.79%, so a topology-independent >90% empirical claim is not supported. Ring-random n=10000/20000 do exceed 90%; the preferential case gives 91.76%.

Proxy pruning is deterministic for exact arithmetic and certified enclosures; these measurements use floating-point triangle bounds inflated by 1e-12 and verify sequence equality against recomputation. They are measurements of fixed-sketch greedy equivalence, not a numerical proof of error enclosure for arbitrary executions.
