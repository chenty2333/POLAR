"""CPU heat-trace reduction for effect selections plus the CPU exact greedy references (no GPU).
dH(t) = Tr exp(-t L0) - Tr exp(-t L_S) on the same log grid as downstream.py.
Usage: python downstream_cpu.py GRAPH POOL(uniform|2hop)  -> results/downstream_<graph>[_2hop]_cpu.json"""
import sys, json, zlib, time
from pathlib import Path
import numpy as np, scipy.linalg as sl
import graphs

T = np.logspace(-2, 2, 33)
name, pool = sys.argv[1], sys.argv[2]; suf = '' if pool == 'uniform' else '_2hop'
n, E, meta = graphs.load(name); seed = 20270000 + zlib.crc32(name.encode()) % 10000
cand = graphs.candidates_2hop(n, E, 2000, seed) if pool == '2hop' else graphs.candidates(n, E, 2000, seed)
sel = {}
e = json.loads(Path(f'results/effect_{name}{suf}.json').read_text())
sel.update({l: v['sequence'] for l, v in e['selections'].items()})
c = json.loads(Path(f'results/exactcpu_{name}{suf}.json').read_text())
sel.update({l: v['sequence'] for l, v in c['runs'].items()})
L0 = np.zeros((n, n)); np.add.at(L0, (E[:, 0], E[:, 1]), -1.0); np.add.at(L0, (E[:, 1], E[:, 0]), -1.0)
L0[np.arange(n), np.arange(n)] = -L0.sum(1)


def heat(L):
    lam = np.maximum(sl.eigvalsh(L, driver='evd', check_finite=False), 0)
    return np.exp(-T[:, None] * lam[None]).sum(1)


out = Path(f'results/downstream_{name}{suf}_cpu.json')
res = json.loads(out.read_text()) if out.exists() else dict(graph=dict(meta, pool=pool), t=T.tolist(), rows={})
h0 = np.array(res['base']) if 'base' in res else heat(L0); res['base'] = h0.tolist()
for label, seq in sel.items():
    if label in res['rows']:
        continue
    t0 = time.time(); L = L0.copy()
    for k in seq:
        a, b = int(cand[k, 0]), int(cand[k, 1]); L[a, a] += 1; L[b, b] += 1; L[a, b] -= 1; L[b, a] -= 1
    res['rows'][label] = dict(heat_reduction=(h0 - heat(L)).tolist())
    out.write_text(json.dumps(res, indent=1)); print(name, pool, label, f'{time.time() - t0:.0f}s', flush=True)
print(name, pool, 'DOWNSTREAM CPU DONE', flush=True)
