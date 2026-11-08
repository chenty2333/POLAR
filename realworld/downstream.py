"""Independent downstream metrics for selections stored in results/effect_*.json (dense GPU eigensolves).

Heat-trace reduction  dH(t) = Tr exp(-t L_0) - Tr exp(-t L_S)  on a log grid of diffusion times t.
By the Mellin identity x^{-s} = Gamma(s)^{-1} int_0^inf t^{s-1} e^{-tx} dt,
    f_s(S) = Gamma(s)^{-1} int_0^inf t^{s-1} e^{-sigma t} dH(t) dt,
so each single t is a different objective, and s sets the horizon distribution Gamma(s, 1/sigma).
Also reports natural connectivity  ln( (1/n) sum_i exp(mu_i(A)) )  of the adjacency matrix.
Usage: python downstream.py results/effect_<graph>[_2hop].json ...
"""
import sys, json
import numpy as np
import cupy as cp
import graphs, run_real

T = np.logspace(-2, 2, 33)


def lap(n, E):
    L = np.zeros((n, n)); np.add.at(L, (E[:, 0], E[:, 1]), -1.0); np.add.at(L, (E[:, 1], E[:, 0]), -1.0)
    L[np.arange(n), np.arange(n)] = -L.sum(1); return L


def metrics(Ld):
    lam = cp.linalg.eigvalsh(Ld); n = Ld.shape[0]
    mu = cp.linalg.eigvalsh(cp.diag(cp.diag(Ld)) - Ld)
    heat = cp.asnumpy(cp.sum(cp.exp(-cp.asarray(T)[:, None] * cp.maximum(lam, 0)[None]), 1))
    m = float(mu.max()); nat = m + float(cp.log(cp.mean(cp.exp(mu - m))))
    return heat, nat


for path in sys.argv[1:]:
    d = json.load(open(path)); name = d['graph']['name']; pool = d['graph'].get('pool', 'uniform')
    run_real.POOL = pool
    G, cand, _ = run_real.setup(name, d['graph']['N'])
    n, E, _ = graphs.load(name); L0 = cp.asarray(lap(n, E))
    h0, nat0 = metrics(L0); out = dict(graph=d['graph'], t=T.tolist(), rows={})
    for label, sel in d['selections'].items():
        L = L0.copy()
        for e in sel['sequence']:
            a, b = int(cand[e, 0]), int(cand[e, 1]); L[a, a] += 1; L[b, b] += 1; L[a, b] -= 1; L[b, a] -= 1
        h, nat = metrics(L)
        out['rows'][label] = dict(heat_reduction=(h0 - h).tolist(), natural_connectivity_gain=nat - nat0)
        print(name, label, f'nat+={nat - nat0:.4e}', flush=True)
    Path = path.replace('effect_', 'downstream_'); open(Path, 'w').write(json.dumps(out, indent=1))
