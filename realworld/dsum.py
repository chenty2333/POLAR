"""Summarize downstream_*.json: heat-trace reduction ratio to the best selection at selected t, and natural connectivity."""
import sys, json, numpy as np
TS = [0.03, 0.1, 0.3, 1, 3, 10, 30]
SHOW = ['Exact(s=0.25)', 'Exact(s=0.5)', 'Exact(s=0.75)', 'Exact-ER', 'Exact-Logdet',
        'POLAR(s=0.25)', 'POLAR(s=0.5)', 'POLAR(s=0.75)', 'Sketched-ER', 'Degree', 'SLQ(s=0.5)']
for p in sys.argv[1:]:
    d = json.load(open(p)); t = np.array(d['t']); R = d['rows']
    lab = [l for l in R if not l.startswith('Random')]
    H = np.array([R[l]['heat_reduction'] for l in lab]); best = H.max(0)
    idx = [int(np.argmin(abs(t - x))) for x in TS]
    print(f"== {d['graph']['name']} pool={d['graph'].get('pool')}   t: " + ' '.join(f'{t[i]:7.2f}' for i in idx) + '    natconn')
    nb = max(R[l]['natural_connectivity_gain'] for l in lab)
    for l in [x for x in SHOW if x in R]:
        print(f'   {l:14s}      ' + ' '.join(f'{H[lab.index(l), i] / best[i]:7.4f}' for i in idx) + f"    {R[l]['natural_connectivity_gain'] / nb:.4f}")
    win = [lab[int(np.argmax(H[:, i]))] for i in idx]
    print('   argmax:', ' | '.join(f'{t[i]:.2f}:{w}' for i, w in zip(idx, win)))
