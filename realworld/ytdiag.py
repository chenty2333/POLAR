"""com-Youtube: is POLAR(s=0.5)'s 1.4% gap to Sketched-ER on f_0.5 a shortlist (B) or sketch-noise (q) effect?"""
import json, time
from pathlib import Path
import numpy as np
import polar_gpu as pg, run_real
G, cand, meta = run_real.setup('com-Youtube', 2000)
e = json.loads(Path('results/effect_com-Youtube.json').read_text())
ref = {l: v[-1] for l, v in e['gains']['frac(s=0.5)'].items() if not l.startswith('_')}
out = Path('results/ytdiag_com-Youtube.json'); res = dict(ref=ref, rows=[])
for q, B in ((128, 64), (256, 16)):
    sk = pg.sketches(G, cand, q, 2026)
    o = pg.polar(G, cand, 20, 0.5, q=q, B=B, sketch=sk)
    g, cert = pg.true_marginals(G, cand, o['sequence'], 0.5)
    row = dict(q=q, B=B, seconds=o['seconds'], gain=float(np.sum(g)), sequence=o['sequence'])
    res['rows'].append(row); out.write_text(json.dumps(res, indent=1))
    best = max(ref.values())
    print(f"q={q} B={B} {o['seconds']:.0f}s  gain/best={row['gain'] / best:.4f}  (POLAR q128B16={ref['POLAR(s=0.5)'] / best:.4f}, Sketched-ER=1.0000)", flush=True)
print('YTDIAG DONE', flush=True)
