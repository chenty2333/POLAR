"""Correctness checks for polar_gpu on a small real graph against dense linear algebra."""
import sys, json, time
import numpy as np
import cupy as cp
import polar_gpu as pg
import graphs

name = sys.argv[1] if len(sys.argv) > 1 else 'email-Eu-core'
n, E, meta = graphs.load(name); sigma = 1.0; k = 10; s = 0.5
G = pg.Graph(n, E, sigma); cand = graphs.candidates(n, E, 500, 7)
Md = G.M0.get().toarray(); lam0 = np.linalg.eigvalsh(Md)
report = {'graph': meta, 'checks': {}}

# 1. multi-shift CG vs dense solves on all active rows
tau, om, _ = pg.composite(s, sigma, G.Lam(k), 1e-2)
w = np.zeros(n); w[cand[0]] = [1, -1]
active = np.unique(cand)
X, h, it = pg.msolve(pg.Op(G), cp.asarray(w)[:, None], tau, cp.asarray(active), 1e-10, track=True)
err_x = err_h = 0
for j in range(0, len(tau), max(1, len(tau) // 8)):
    u = np.linalg.solve(Md + tau[j] * np.eye(n), w)
    err_x = max(err_x, np.abs(u[active] - X[j, :, 0].get()).max() / np.abs(u).max())
    err_h = max(err_h, abs(u @ u - float(h[j, 0])) / (u @ u))
report['checks']['msolve'] = dict(p=len(tau), iters=int(it[0]), rel_err_rows=err_x, rel_err_tracked_norm=err_h)

# 2. certified true marginals vs eigenvalue marginals along a random sequence
seq = pg.random_set(cand, k, 3)
g, cert = pg.true_marginals(G, cand, seq, s)
A = Md.copy(); prev = np.sum(lam0 ** -s); rel = 0
for e, ge in zip(seq, g):
    a, b = cand[e]; A[a, a] += 1; A[b, b] += 1; A[a, b] -= 1; A[b, a] -= 1
    cur = np.sum(np.linalg.eigvalsh(A) ** -s); rel = max(rel, abs((prev - cur) - ge) / (prev - cur)); prev = cur
report['checks']['true_marginals_frac'] = dict(max_rel_err=rel, **cert)
for obj in ('er', 'logdet'):
    g2, _ = pg.true_marginals(G, cand, seq, s, obj=obj)
    A = Md.copy(); rel = 0
    for e, ge in zip(seq, g2):
        a, b = cand[e]; bvec = np.zeros(n); bvec[a] = 1; bvec[b] = -1
        Ri = np.linalg.solve(A, bvec); d = bvec @ Ri
        ex = (Ri @ Ri) / (1 + d) if obj == 'er' else np.log1p(d)
        rel = max(rel, abs(ex - ge) / ex)
        A[a, a] += 1; A[b, b] += 1; A[a, b] -= 1; A[b, a] -= 1
    report['checks'][f'true_marginals_{obj}'] = dict(max_rel_err=rel)

# 3. maintained POLAR (B=1) reproduces recomputed Naive sequences
sk = pg.sketches(G, cand, 24, 2026)
for obj in ('frac', 'er', 'logdet'):
    a1 = pg.polar(G, cand, k, s, B=1, obj=obj, sketch=sk)
    a0 = pg.polar(G, cand, k, s, obj=obj, sketch=sk, naive=True)
    a4 = pg.polar(G, cand, k, s, B=4, obj=obj, sketch=sk)
    report['checks'][f'naive_vs_polar_{obj}'] = dict(match=sum(x == y for x, y in zip(a0['sequence'], a1['sequence'])),
        naive_spmv=a0['spmv'], polar_spmv=a1['spmv'], naive_s=a0['seconds'], polar_s=a1['seconds'], polarB4_s=a4['seconds'], p=a1['p'])
print(json.dumps(report, indent=1, default=float))
