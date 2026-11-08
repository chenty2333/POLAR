# Real-network experiments

Requirements: Python 3.12, NumPy, SciPy, pandas, and CuPy 14 with an NVIDIA
GPU (we used one RTX 4090, 24 GB). The CPU exact references
(`exact_cpu.py`, `downstream_cpu.py`) need only NumPy/SciPy.

| File | Purpose |
|---|---|
| `polar_gpu.py` | POLAR (cache backend, multi-shift CG, shortlist re-ranking), Naive, Degree, Random, certified marginal evaluation |
| `spmm.py` | warp-per-row CSR kernel for multi-vector SpMM |
| `graphs.py` | loading/preprocessing of SNAP and KONECT inputs; uniform and 2-hop candidate pools |
| `exact.py`, `exact_cpu.py` | exact greedy with certified interval screening and the twin closed form (GPU / CPU LAPACK) |
| `slq.py` | stochastic Lanczos quadrature greedy baseline |
| `run_real.py` | drivers: `effect` (quality), `time` (Table I), `opt` (exhaustive search), `param` (Table V) |
| `diag2.py` | sketch grid, per-round recovery, and sigma sweep |
| `downstream_cpu.py` | heat-trace reduction (Fig. 2) |
| `report.py` | rebuilds `REPORT_realworld.md` from `results/` |
| `*.sh` | the batch scripts used for the runs |

Inputs are read from `$POLAR_DATA/raw` (default `realworld/data/raw`);
download and verify all of them with
`python fetch.py data/raw $(cat sources.txt)`. Preprocessing
keeps all endpoint vertices, drops directions, duplicates, and self-loops.

| Network | n | m | Source file | SHA-256 |
|---|---:|---:|---|---|
| Zachary | 34 | 78 | KONECT `download.tsv.ucidata-zachary.tar.bz2` | `37ad42a41e95a09c1d1e5de5ce9eae3a53d8170c6c4af6c4a0121885cafc2cef` |
| Dolphins | 62 | 159 | KONECT `download.tsv.dolphins.tar.bz2` | `30fb299678fcb137e7caacd795a68490953a38046cb617597a308bc08c0ed811` |
| LesMis | 77 | 254 | KONECT `download.tsv.moreno_lesmis.tar.bz2` | `04f8ed2c517fbda9c60d244041837a08328ee13c3102efe847e911dc97a27fbe` |
| Polbooks | 105 | 441 | KONECT `download.tsv.dimacs10-polbooks.tar.bz2` | `f93e32e71bb43c2bc0305236b0049a867c6c0bc466a9984ac9e72c69b6323acb` |
| Hamster | 921 | 4,032 | KONECT `download.tsv.petster-hamster-household.tar.bz2` | `6c9961e50fa10e6f3b12077442d27c5c9ee12f089f2aabec05516945ea0f738a` |
| email-Eu-core | 1,005 | 16,064 | SNAP `email-Eu-core.txt.gz` | `4b47acdb80197b085fe63c819c357ae488131ee904ed93d1b219a68b0f9e245f` |
| ca-GrQc | 5,242 | 14,484 | SNAP `ca-GrQc.txt.gz` | `a254442cdf5d684712578b630c2e0d7543518ab154ef2341cabb607572ce7230` |
| ca-HepPh | 12,008 | 118,489 | SNAP `ca-HepPh.txt.gz` | `8d679f64ea507834613f4c09e6f692cc8a2405d5a01bfe1ee3b24fbbea1d807f` |
| ca-CondMat | 23,133 | 93,439 | SNAP `ca-CondMat.txt.gz` | `11b47db52441ce0140f2ec54999345459fecbc52ff6f9fe3956dc3cad9835872` |
| email-Enron | 36,692 | 183,831 | SNAP `email-Enron.txt.gz` | `55cfead79b1f0161786179a48796c2a119bd7026a246238b257b6be9d8b69b68` |
| com-Amazon | 334,863 | 925,872 | SNAP `com-amazon.ungraph.txt.gz` | `bb4b9fcf549c9174049e885e9a0a4057c9724846e7297c4a1ea299b8e92eec34` |
| com-DBLP | 317,080 | 1,049,866 | SNAP `com-dblp.ungraph.txt.gz` | `9eb0bd30312ddd04e2624f7c36c0983a2e99b116f0385be5a7fce6d6170f4cb3` |
| roadNet-PA | 1,088,092 | 1,541,898 | SNAP `roadNet-PA.txt.gz` | `450b8733635d887466a2b96b26411f6e62caf7006f8a264f59cf9b5d75cdf549` |
| roadNet-CA | 1,965,206 | 2,766,607 | SNAP `roadNet-CA.txt.gz` | `383f7b14424530a9e25c392969a9c19dace32ebcfd141f0967a1433498c2acdd` |
| com-Youtube | 1,134,890 | 2,987,624 | SNAP `com-youtube.ungraph.txt.gz` | `dff1b97ba7d2fa9c59884b67dcd2275e717ff9501f86ed82ce6582ed4971f3e0` |
| web-Google | 875,713 | 4,322,051 | SNAP `web-Google.txt.gz` | `bcac0af0471d749f4a8c010bca92b61cf2868a0570741de06892fc062f265ea6` |
| as-Skitter | 1,696,415 | 11,095,298 | SNAP `as-skitter.txt.gz` | `743d4f91fbcbcace4a70d44d1190fe75869faa1398d3eb73bb2d7663c2417510` |
| com-LiveJournal | 3,997,962 | 34,681,189 | SNAP `com-lj.ungraph.txt.gz` | `e0e8996341b963ecbe0a572c71446c03fd6e93a6c521578eaae7566c68ff608e` |

The same hashes are stored in the `graph` metadata of every result JSON.

Examples (defaults: sigma = 1, k = 20, N = 2000, q = 128, B = 16,
eps = 1e-2, CG tolerance 1e-7, projection seed 2026):

    python run_real.py time   --graphs com-DBLP --s 0.5          # Table I
    python run_real.py effect --graphs ca-GrQc --slq             # Table II
    python run_real.py effect --graphs ca-GrQc --pool 2hop       # Table II, 2-hop
    python run_real.py opt    --graphs Zachary --N 40 --k 5 --s 0.5
    python run_real.py param  --graphs ca-GrQc --s 0.5           # Table V
    python exact_cpu.py ...                                      # see cpujobs*.sh
    python report.py                                             # all tables

## Results (`results/`)

| Pattern | Content |
|---|---|
| `time_*.json` | Table I: all timing repetitions, SpMV and right-hand-side counts, selections |
| `effect_*.json` | selections and prefix gains of every method on every target (`_2hop`: 2-hop pool, `_ps1/_ps2`: other uniform pools, `_k100`) |
| `exactcpu_*.json` | CPU exact greedy references for ca-HepPh and ca-CondMat |
| `opt_*.json` | exhaustive optima on the small networks |
| `param_ca-GrQc.json` | Table V |
| `diag_*.json` | sketch grid, per-round recovery, sigma sweep |
| `downstream_*.json` | heat-trace reductions (Fig. 2) |
| `ytdiag_com-Youtube.json` | com-Youtube shortlist/sketch-size diagnosis |
| `logs/` | run logs |

`REPORT_realworld.md` summarizes every result in Markdown tables.
