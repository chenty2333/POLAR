"""Real-network experiments for POLAR.

  effect : selection + certified true gains for all methods, every prefix r=1..k, s in S_LIST,
           plus the exponent cross-evaluation (each POLAR(s') sequence evaluated under every s).
  time   : Naive vs POLAR running time / SpMVs, median of R repetitions with rotated order.
  opt    : brute-force optimum on tiny graphs vs every method.
  param  : q / B / eps sensitivity on one graph.
Results: results/<mode>_<graph>.json
"""
import argparse, json, time, platform, zlib
from pathlib import Path
import numpy as np
import cupy as cp
import polar_gpu as pg
import graphs
import exact

OUT = Path(__file__).resolve().parent / 'results'
SUF = ''
SIGMA = 1.0


def env():
    return dict(gpu=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(), cupy=cp.__version__,
                numpy=np.__version__, python=platform.python_version(), host=platform.node())


POOL = 'uniform'
PSEED = 0           # candidate-pool seed offset (robustness runs)


def setup(name, N, seed_offset=None):
    seed_offset = PSEED if seed_offset is None else seed_offset
    n, E, meta = graphs.load(name)
    G = pg.Graph(n, E, SIGMA)
    seed = 20270000 + zlib.crc32(name.encode()) % 10000 + seed_offset
    if not N:
        cand = graphs.all_nonedges(n, E)
    elif POOL == '2hop':
        cand = graphs.candidates_2hop(n, E, N, seed)
    else:
        cand = graphs.candidates(n, E, N, seed)
    meta.update(N=len(cand), candidate_seed=seed, pool=POOL, dmax=G.dmax, sigma=SIGMA,
                n_active=int(len(np.unique(cand))))
    return G, cand, meta


def dense(G):
    return G.M0.get().toarray()


def evaluate(G, cand, seq, s, obj, use_dense, Md):
    if use_dense:
        return exact.eval_prefix(Md, cand, seq, s, obj).tolist(), dict(evaluator='eigenvalues')
    g, cert = pg.true_marginals(G, cand, seq, s, obj=obj)
    return np.cumsum(g).tolist(), dict(evaluator='certified-quadrature', **cert)


def effect(a):
    for name in a.graphs:
        G, cand, meta = setup(name, a.N); k = a.k
        use_dense = G.n <= a.dense_max; Md = dense(G) if use_dense else None
        sk = pg.sketches(G, cand, a.q, 2026)
        res = dict(mode='effect', graph=meta, params=dict(k=k, q=a.q, B=a.B, eps=a.eps, tol=a.tol, s=a.s),
                   env=env(), selections={}, gains={})
        sel = res['selections']
        t = time.perf_counter()

        def add(label, out):
            sel[label] = out; print(name, label, f"{out.get('seconds', 0):.2f}s", flush=True)
        for s in a.s:
            add(f'POLAR(s={s})', pg.polar(G, cand, k, s, q=a.q, B=a.B, eps=a.eps, tol=a.tol, sketch=sk))
        add('Sketched-ER', pg.polar(G, cand, k, 1.0, q=a.q, B=a.B, tol=a.tol, obj='er', sketch=sk))
        add('Sketched-Logdet', pg.polar(G, cand, k, 0.5, q=a.q, B=a.B, tol=a.tol, obj='logdet', sketch=sk))
        add('Degree', dict(sequence=pg.degree_greedy(G, cand, k)))
        if a.slq:
            import slq
            for s in a.s:
                add(f'SLQ(s={s})', slq.slq_greedy(G, cand, k, s, probes=a.slq_probes, m=a.slq_steps))
        for r in range(a.random):
            add(f'Random#{r}', dict(sequence=pg.random_set(cand, k, 1000 + r)))
        if use_dense:
            for s in a.s:
                add(f'Exact(s={s})', exact.exact_greedy(Md, cand, k, s, 'frac'))
            add('Exact-ER', exact.exact_greedy(Md, cand, k, 1.0, 'er'))
            add('Exact-Logdet', exact.exact_greedy(Md, cand, k, 0.5, 'logdet'))
        targets = [('frac', s) for s in a.s] + [('er', 1.0), ('logdet', 0.0)]
        keys = [f'{o}' if o != 'frac' else f'frac(s={s})' for o, s in targets]
        for key in keys:
            res['gains'][key] = {}
        if use_dense:
            for label, out in sel.items():
                vals = exact.eval_prefix_multi(Md, cand, out['sequence'], targets)
                for key, v in zip(keys, vals):
                    res['gains'][key][label] = v.tolist()
            for key in keys:
                res['gains'][key]['_evaluator'] = dict(evaluator='eigenvalues')
        else:
            for (obj, s), key in zip(targets, keys):
                if obj != 'frac' and a.cross_only:
                    labels = [l for l in sel if not l.startswith('Random')]
                else:
                    labels = list(sel)
                for label in labels:
                    g, info = evaluate(G, cand, sel[label]['sequence'], s, obj, False, None)
                    res['gains'][key][label] = g
                res['gains'][key]['_evaluator'] = info
                print(name, 'evaluated', key, flush=True)
                OUT.mkdir(exist_ok=True); (OUT / f'effect_{name}{SUF}.json').write_text(json.dumps(res, indent=1))
        res['seconds_total'] = time.perf_counter() - t
        OUT.mkdir(exist_ok=True); (OUT / f'effect_{name}{SUF}.json').write_text(json.dumps(res, indent=1))


def timing(a):
    for name in a.graphs:
        G, cand, meta = setup(name, a.N); sk = pg.sketches(G, cand, a.q, 2026)
        res = dict(mode='time', graph=meta, params=dict(k=a.k, q=a.q, B=a.B, eps=a.eps, tol=a.tol, s=a.s[0],
                   repetitions=a.reps), env=env(), runs={})
        cfg = {'Naive': dict(naive=True), 'POLAR(B=1)': dict(B=1), f'POLAR(B={a.B})': dict(B=a.B)}
        if a.skip_naive:
            cfg.pop('Naive')
        names = list(cfg)
        reps = a.reps if meta['m'] < 2e7 else 1          # one repetition on graphs with > 2e7 edges (LiveJournal)
        res['params'].update(naive_reps=a.naive_reps, naive_limit=a.naive_limit, repetitions=reps)
        for rep in range(reps):
            for nm in names[rep % len(names):] + names[:rep % len(names)]:
                if nm == 'Naive' and rep >= a.naive_reps:
                    continue
                lim = a.naive_limit if nm == 'Naive' else None
                out = pg.polar(G, cand, a.k, a.s[0], q=a.q, eps=a.eps, tol=a.tol, sketch=sk, time_limit=lim, **{'B': a.B, **cfg[nm]})
                res['runs'].setdefault(nm, []).append(out)
                print(name, nm, rep, f"{out['seconds']:.2f}s spmv={out['spmv']}", flush=True)
                OUT.mkdir(exist_ok=True); (OUT / f'time_{name}{SUF}.json').write_text(json.dumps(res, indent=1))


def optimum(a):
    for name in a.graphs:
        G, cand, meta = setup(name, a.N); Md = dense(G); s = a.s[0]
        res = dict(mode='opt', graph=meta, s=s, env=env(), rows=[])
        sk = pg.sketches(G, cand, a.q, 2026)
        for k in range(1, a.k + 1):
            t = time.perf_counter(); opt, arg = exact.brute_force(Md, cand, k, s); tb = time.perf_counter() - t
            row = dict(k=k, opt=opt, opt_set=arg, brute_seconds=tb, methods={})
            seqs = {'POLAR': pg.polar(G, cand, k, s, q=a.q, B=a.B, eps=a.eps, tol=a.tol, sketch=sk)['sequence'],
                    'Exact': exact.exact_greedy(Md, cand, k, s)['sequence'],
                    'Exact-ER': exact.exact_greedy(Md, cand, k, 1.0, 'er')['sequence'],
                    'Exact-Logdet': exact.exact_greedy(Md, cand, k, 0.5, 'logdet')['sequence'],
                    'Degree': pg.degree_greedy(G, cand, k)}
            for r in range(a.random):
                seqs[f'Random#{r}'] = pg.random_set(cand, k, 1000 + r)
            for m, sq in seqs.items():
                row['methods'][m] = float(exact.eval_prefix(Md, cand, sq, s)[-1])
            res['rows'].append(row); print(name, k, opt, {m: round(v / opt, 4) for m, v in row['methods'].items() if '#' not in m}, flush=True)
            OUT.mkdir(exist_ok=True); (OUT / f'opt_{name}_s{s}{SUF}.json').write_text(json.dumps(res, indent=1))


def param(a):
    for name in a.graphs:
        G, cand, meta = setup(name, a.N); s = a.s[0]; use_dense = G.n <= a.dense_max; Md = dense(G) if use_dense else None
        res = dict(mode='param', graph=meta, s=s, env=env(), rows=[])
        ref = exact.exact_greedy(Md, cand, a.k, s)['sequence'] if use_dense else None
        refg = evaluate(G, cand, ref, s, 'frac', use_dense, Md)[0][-1] if ref else None
        grid = [dict(q=q, B=a.B, eps=a.eps) for q in (16, 32, 64, 96, 128, 192, 256)]
        grid += [dict(q=a.q, B=B, eps=a.eps) for B in (1, 2, 4, 8, 32, 64)]
        grid += [dict(q=a.q, B=a.B, eps=e) for e in (1e-1, 1e-3, 1e-4)]
        for g in grid:
            out = pg.polar(G, cand, a.k, s, q=g['q'], B=g['B'], eps=g['eps'], tol=a.tol, seed=2026)
            gain = evaluate(G, cand, out['sequence'], s, 'frac', use_dense, Md)[0][-1]
            row = dict(**g, gain=gain, ref_gain=refg, seconds=out['seconds'], spmv=out['spmv'], p=out['p'])
            res['rows'].append(row); print(name, row, flush=True)
            OUT.mkdir(exist_ok=True); (OUT / f'param_{name}{SUF}.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    P = argparse.ArgumentParser()
    P.add_argument('mode', choices=['effect', 'time', 'opt', 'param'])
    P.add_argument('--graphs', nargs='+', required=True)
    P.add_argument('--N', type=int, default=2000); P.add_argument('--k', type=int, default=20)
    P.add_argument('--q', type=int, default=128); P.add_argument('--B', type=int, default=16)
    P.add_argument('--eps', type=float, default=1e-2); P.add_argument('--tol', type=float, default=1e-7)
    P.add_argument('--s', type=float, nargs='+', default=[0.25, 0.5, 0.75])
    P.add_argument('--random', type=int, default=10); P.add_argument('--reps', type=int, default=3)
    P.add_argument('--naive-reps', type=int, default=1); P.add_argument('--naive-limit', type=float, default=3600.0)
    P.add_argument('--dense-max', type=int, default=6000); P.add_argument('--skip-naive', action='store_true')
    P.add_argument('--cross-only', action='store_true', help='evaluate Random only under the fractional targets')
    P.add_argument('--pool', choices=['uniform', '2hop'], default='uniform')
    P.add_argument('--slq', action='store_true'); P.add_argument('--slq-probes', type=int, default=20)
    P.add_argument('--slq-steps', type=int, default=40); P.add_argument('--tag', default=''); P.add_argument('--pool-seed', type=int, default=0)
    a = P.parse_args()
    POOL = a.pool; PSEED = a.pool_seed
    SUF = ('' if a.pool == 'uniform' else '_2hop') + (f'_ps{a.pool_seed}' if a.pool_seed else '') + (f'_{a.tag}' if a.tag else '')
    {'effect': effect, 'time': timing, 'opt': optimum, 'param': param}[a.mode](a)
