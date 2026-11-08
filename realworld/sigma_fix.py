"""Recompute only the POLAR(s=0.5) row of results/diag_sigma_*.json with q=128, B=16
(the original sigma sweep called polar() with the old q=24, B=4 defaults)."""
import sys, json
from pathlib import Path
import exact, polar_gpu as pg, run_real
k = 20
for name in sys.argv[1:]:
    p = Path('results') / f'diag_sigma_{name}.json'; d = json.loads(p.read_text())
    for row in d['rows']:
        run_real.SIGMA = row['sigma']; G2, cand2, _ = run_real.setup(name, 2000); M2 = G2.M0.get().toarray()
        targets = [(t.split(':')[0], float(t.split(':')[1])) for t in row['targets']]
        sq = pg.polar(G2, cand2, k, 0.5, q=128, B=16)['sequence']
        er = set(exact.exact_greedy(M2, cand2, k, 1.0, 'er')['sequence'])
        row['final_gain']['POLAR(s=0.5)'] = [float(v[-1]) for v in exact.eval_prefix_multi(M2, cand2, sq, targets)]
        row['jaccard']['POLAR(s=0.5)'] = len(set(sq) & er) / len(set(sq) | er)
        F = row['final_gain']; ex = F['Exact(s=0.5)']
        print(name, row['sigma'], 'POLAR/Exact(s=0.5) on f_0.5 = %.4f' % (F['POLAR(s=0.5)'][1] / ex[1]), flush=True)
    p.write_text(json.dumps(d, indent=1))
