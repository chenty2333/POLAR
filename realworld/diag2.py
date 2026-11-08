"""Selection diagnostics on medium graphs (dense references).
grid  : quality over q x B x sketch kind x 5 sketch seeds (fixed candidate pool).
round : along POLAR's own trajectory, r_t = g(chosen|S_t)/max_e g(e|S_t) and
        shortlist_t = max_{e in shortlist} g(e|S_t)/max_e g(e|S_t), with exact marginals.
sigma : exponent cross-evaluation of exact greedy for sigma in {0.01, 0.1, 1}.
"""
import sys, json, numpy as np, cupy as cp
import polar_gpu as pg, run_real, exact, graphs
from pathlib import Path
OUT = Path('results'); OUT.mkdir(exist_ok=True)
mode, names = sys.argv[1], sys.argv[2:]; k = 20; s = 0.5


def exact_marginals(M, cand, s, live):
    lam, V = cp.linalg.eigh(M); W = (V[cp.asarray(cand[:, 0])] - V[cp.asarray(cand[:, 1])]) ** 2
    tau, om, _ = pg.composite(s, float(lam[0]) * 0.999, float(lam[-1]) + 2.001, 1e-9)
    D = 1 / (lam[:, None] + cp.asarray(tau)[None]); g = cp.asnumpy(((W @ (D * D)) / (1 + W @ D)) @ cp.asarray(om))
    g[~live] = -np.inf; return g


for name in names:
    G, cand, meta = run_real.setup(name, 2000); Md = G.M0.get().toarray()
    if mode == 'grid':
        ref = exact.exact_greedy(Md, cand, k, s)['sequence']; refg = exact.eval_prefix(Md, cand, ref, s)[-1]; rows = []
        for kind in ('block', 'gauss'):
            for q in (24, 64, 128):
                for seed in range(5):
                    sk = pg.sketches(G, cand, q, 3000 + seed, kind)
                    for B in (4, 16, 32):
                        o = pg.polar(G, cand, k, s, q=q, B=B, sketch=sk)
                        r = exact.eval_prefix(Md, cand, o['sequence'], s)[-1] / refg
                        rows.append(dict(kind=kind, q=q, B=B, seed=seed, ratio=r, seconds=o['seconds'], spmv=o['spmv']))
                        print(name, kind, q, B, seed, round(r, 4), flush=True)
        (OUT / f'diag_grid_{name}.json').write_text(json.dumps(dict(graph=meta, ref_gain=refg, rows=rows), indent=1))
    elif mode == 'round':
        res = []
        for kind, q, B in (('block', 24, 4), ('block', 128, 16), ('gauss', 128, 16)):
            for seed in range(5):
                tr = []; o = pg.polar(G, cand, k, s, q=q, B=B, sketch=pg.sketches(G, cand, q, 3000 + seed, kind), trace=tr)
                M = cp.asarray(Md); live = np.ones(len(cand), bool); rt = []; st = []
                for step in tr:
                    g = exact_marginals(M, cand, s, live); best = g.max()
                    rt.append(float(g[step['chosen']] / best)); st.append(float(max(g[step['shortlist']]) / best))
                    e = step['chosen']; live[e] = False; a, b = int(cand[e, 0]), int(cand[e, 1])
                    M[a, a] += 1; M[b, b] += 1; M[a, b] -= 1; M[b, a] -= 1
                res.append(dict(kind=kind, q=q, B=B, seed=seed, r=rt, shortlist=st))
                print(name, kind, q, B, seed, 'min r %.4f mean r %.4f min shortlist %.4f' % (min(rt), np.mean(rt), min(st)), flush=True)
        (OUT / f'diag_round_{name}.json').write_text(json.dumps(dict(graph=meta, rows=res), indent=1))
    elif mode == 'sigma':
        n, E, _ = graphs.load(name); rows = []
        for sig in (0.01, 0.1, 1.0):
            run_real.SIGMA = sig; G2, cand2, meta2 = run_real.setup(name, 2000); M2 = G2.M0.get().toarray()
            targets = [('frac', 0.25), ('frac', 0.5), ('frac', 0.75), ('er', 1.0), ('logdet', 0.0)]
            seqs = {f'Exact(s={x})': exact.exact_greedy(M2, cand2, k, x)['sequence'] for x in (0.25, 0.5, 0.75)}
            seqs['Exact-ER'] = exact.exact_greedy(M2, cand2, k, 1.0, 'er')['sequence']
            seqs['Exact-Logdet'] = exact.exact_greedy(M2, cand2, k, 0.5, 'logdet')['sequence']
            seqs['Degree'] = pg.degree_greedy(G2, cand2, k)
            o = pg.polar(G2, cand2, k, 0.5, q=128, B=16); seqs['POLAR(s=0.5)'] = o['sequence']
            tab = {m: [float(v[-1]) for v in exact.eval_prefix_multi(M2, cand2, sq, targets)] for m, sq in seqs.items()}
            rows.append(dict(sigma=sig, targets=[f'{a}:{b}' for a, b in targets], final_gain=tab,
                             jaccard={m: len(set(sq) & set(seqs['Exact-ER'])) / len(set(sq) | set(seqs['Exact-ER'])) for m, sq in seqs.items()}))
            print(name, 'sigma', sig, {m: [round(v / max(tab[x][i] for x in tab), 4) for i, v in enumerate(vals)] for m, vals in tab.items()}, flush=True)
        run_real.SIGMA = 1.0
        (OUT / f'diag_sigma_{name}.json').write_text(json.dumps(dict(rows=rows), indent=1))
