"""Check the twin closed form against eigvalsh, and the new exact greedy against stored sequences."""
import sys, json, time, numpy as np, cupy as cp
import exact, run_real
# 1) closed form vs eigenvalues on ca-GrQc 2-hop twins
run_real.POOL = '2hop'; G, cand, _ = run_real.setup('ca-GrQc', 2000); M = cp.asarray(G.M0.get().toarray())
for s in (0.25, 0.5, 0.75):
    tw = exact.twin_gain(M, cand, s); ids = np.flatnonzero(np.isfinite(tw)); old = float(cp.sum(cp.linalg.eigvalsh(M) ** -s))
    err = []
    for c in ids[:6]:
        A = M.copy(); exact._add(A, int(cand[c, 0]), int(cand[c, 1]))
        err.append(abs(old - float(cp.sum(cp.linalg.eigvalsh(A) ** -s)) - tw[c]) / tw[c])
    print('twins', len(ids), 's', s, 'max rel err vs eig %.2e' % max(err), flush=True)
# 2) new exact greedy vs stored Exact(s) sequences
for name, pool in [(g, p) for p in ('uniform', '2hop') for g in sys.argv[1:]]:
    run_real.POOL = pool; G, cand, _ = run_real.setup(name, 2000); Md = G.M0.get().toarray()
    d = json.load(open(f'results/effect_{name}{"_2hop" if pool == "2hop" else ""}.json'))
    for s in (0.25, 0.5, 0.75):
        o = exact.exact_greedy(Md, cand, 20, s); old = d['selections'][f'Exact(s={s})']
        gn, go = exact.eval_prefix(Md, cand, o['sequence'], s)[-1], exact.eval_prefix(Md, cand, old['sequence'], s)[-1]
        print(name, pool, s, 'same seq' if o['sequence'] == old['sequence'] else 'DIFF seq',
              'gain new/old %.12f' % (gn / go), 't %.1fs (old %.1fs)' % (o['seconds'], old.get('seconds', -1)),
              'eig solves', sum(c[1] for c in o['contenders']), flush=True)
