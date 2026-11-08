"""CPU (LAPACK, multithreaded) exact greedy reference for graphs beyond the GPU dense range.
Same rule as exact.exact_greedy: eps=1e-9 composite screening, twin closed form, eigenvalue resolution
of the remaining contenders. Each round's eigenvalues also give exact prefix values of every target.
Uses no GPU. Usage: python exact_cpu.py GRAPH POOL(uniform|2hop) [N] [k] [methods...]
methods: frac:0.5 er logdet frac:0.25 frac:0.75 (default: all, in this order)."""
import sys, json, time, zlib
from pathlib import Path
import numpy as np, scipy.linalg as sl
import graphs
from polar_gpu import composite

SIGMA = 1.0
TARGETS = [('frac', 0.25), ('frac', 0.5), ('frac', 0.75), ('er', 1.0), ('logdet', 0.0)]


def tval(lam, o, s):
    return np.sum(lam ** -s) if o == 'frac' else (np.sum(1 / lam) if o == 'er' else -np.sum(np.log(lam)))


def add(M, a, b):
    M[a, a] += 1; M[b, b] += 1; M[a, b] -= 1; M[b, a] -= 1


def twin_gain(M, pairs, s):
    a, b = pairs[:, 0], pairs[:, 1]; r = np.arange(len(pairs))
    D = M[a] - M[b]; D[r, a] = 0; D[r, b] = 0
    twin = (np.abs(D).max(1) == 0) & (M[a, b] == 0) & (M[a, a] == M[b, b])
    x = M[a, a]; g = -x ** (-s) * np.expm1(-s * np.log1p(2 / x))
    return np.where(twin, g, np.nan)


def greedy(M0, cand, k, s, obj, log):
    M = M0.copy(); live = np.ones(len(cand), bool); seq, cont, prefix = [], [], []
    ea, eb = cand[:, 0], cand[:, 1]; t0 = time.perf_counter()
    for t in range(k):
        lam, V = sl.eigh(M, driver='evd', overwrite_a=False, check_finite=False)
        prefix.append([float(tval(lam, o, x)) for o, x in TARGETS])
        W = (V[ea] - V[eb]) ** 2; del V
        if obj == 'er':
            sc = (W @ lam ** -2) / (1 + W @ (1 / lam)); sc[~live] = -np.inf; e = int(np.argmax(sc)); cont.append((1, 0))
        elif obj == 'logdet':
            sc = np.log1p(W @ (1 / lam)); sc[~live] = -np.inf; e = int(np.argmax(sc)); cont.append((1, 0))
        else:
            tau, om, eps = composite(s, float(lam[0]) * 0.999, float(lam[-1]) + 2.001, 1e-9); eps = max(eps, 1e-9)
            D = 1 / (lam[:, None] + tau[None]); sc = ((W @ (D * D)) / (1 + W @ D)) @ om
            sc[~live] = -np.inf; best_lo = float(sc.max()) / (1 + eps)
            ids = np.flatnonzero((sc / (1 - eps) >= best_lo) & live)
            if len(ids) == 1:
                e = int(ids[0]); cont.append((1, 0))
            else:
                tw = twin_gain(M, cand[ids], s); old = float(np.sum(lam ** -s)); vals = np.empty(len(ids)); solved = 0
                for i, c in enumerate(ids):
                    if np.isfinite(tw[i]):
                        vals[i] = tw[i]; continue
                    A = M.copy(); add(A, int(cand[c, 0]), int(cand[c, 1])); solved += 1
                    vals[i] = old - float(np.sum(sl.eigvalsh(A, driver='evd', overwrite_a=True, check_finite=False) ** -s))
                e = int(ids[int(np.argmax(vals))]); cont.append((len(ids), solved))
        seq.append(e); live[e] = False; add(M, int(cand[e, 0]), int(cand[e, 1]))
        print(f'  round {t + 1}/{k} edge {e} contenders {cont[-1]} {time.perf_counter() - t0:.0f}s', file=log, flush=True)
    lam = sl.eigvalsh(M, driver='evd', check_finite=False); prefix.append([float(tval(lam, o, x)) for o, x in TARGETS])
    P = np.array(prefix); gains = (P[0][None] - P[1:]).T.tolist()     # gains[target][r] = f(S_{r+1})
    return dict(sequence=seq, seconds=time.perf_counter() - t0, contenders=cont, prefix_gains=gains)


if __name__ == '__main__':
    name, pool = sys.argv[1], sys.argv[2]
    N = int(sys.argv[3]) if len(sys.argv) > 3 else 2000; k = int(sys.argv[4]) if len(sys.argv) > 4 else 20
    methods = sys.argv[5:] or ['frac:0.5', 'er', 'logdet', 'frac:0.25', 'frac:0.75']
    n, E, meta = graphs.load(name); seed = 20270000 + zlib.crc32(name.encode()) % 10000
    cand = graphs.candidates_2hop(n, E, N, seed) if pool == '2hop' else graphs.candidates(n, E, N, seed)
    M0 = np.zeros((n, n)); np.add.at(M0, (E[:, 0], E[:, 1]), -1.0); np.add.at(M0, (E[:, 1], E[:, 0]), -1.0)
    M0[np.arange(n), np.arange(n)] = -M0.sum(1) + SIGMA
    out = Path('results') / f'exactcpu_{name}{"" if pool == "uniform" else "_" + pool}{"" if k == 20 else f"_k{k}"}.json'
    res = json.loads(out.read_text()) if out.exists() else dict(graph=dict(meta, N=N, candidate_seed=seed, pool=pool, sigma=SIGMA),
                                                                 k=k, targets=[f'{o}:{x}' for o, x in TARGETS], runs={})
    for m in methods:
        o, s = (m.split(':')[0], float(m.split(':')[1])) if ':' in m else (m, 1.0 if m == 'er' else 0.5)
        label = f'Exact(s={s})' if o == 'frac' else ('Exact-ER' if o == 'er' else 'Exact-Logdet')
        if label in res['runs']:
            continue
        print(name, pool, label, 'start', flush=True)
        res['runs'][label] = greedy(M0, cand, k, s, o, sys.stdout)
        print(name, pool, label, f"{res['runs'][label]['seconds']:.0f}s", flush=True)
        out.write_text(json.dumps(res, indent=1))
    print(name, pool, 'CPU EXACT DONE', flush=True)
