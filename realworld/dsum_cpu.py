import sys, json, numpy as np
TS = [0.03, 0.1, 0.32, 1, 3.16, 10, 31.6]
for p in sys.argv[1:]:
    d = json.load(open(p)); t = np.array(d['t']); R = d['rows']
    lab = [l for l in R if not l.startswith('Random')]; H = np.array([R[l]['heat_reduction'] for l in lab]); best = H.max(0)
    idx = [int(np.argmin(abs(t - x))) for x in TS]
    print(f"== {d['graph']['name']} {d['graph']['pool']}   t: " + ' '.join(f'{t[i]:7.2f}' for i in idx))
    for l in sorted(lab, key=lambda x: (not x.startswith('Exact'), x)):
        print(f'   {l:16s}' + ' '.join(f'{H[lab.index(l), i] / best[i]:7.4f}' for i in idx))
