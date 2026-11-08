"""Print effect-mode summaries: final gains per target, relative to exact (or best) and Degree."""
import json, glob, sys, numpy as np
for f in sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else 'results/effect_*.json')):
    d = json.load(open(f)); g = d['graph']; print(f"== {g['name']} n={g['n']} m={g['m']} N={g['N']} dmax={g['dmax']}")
    for key, tab in d['gains'].items():
        fin = {l: v[-1] for l, v in tab.items() if not l.startswith('_')}
        rnd = [v for l, v in fin.items() if l.startswith('Random')]
        rows = {l: v for l, v in fin.items() if not l.startswith('Random')}
        if rnd: rows['Random(mean)'] = float(np.mean(rnd))
        best = max(rows.values()); ex = {'frac(s=0.25)': 'Exact(s=0.25)', 'frac(s=0.5)': 'Exact(s=0.5)', 'frac(s=0.75)': 'Exact(s=0.75)', 'er': 'Exact-ER', 'logdet': 'Exact-Logdet'}.get(key)
        ref = rows.get(ex, best)
        print(f"  [{key}] ref={'exact' if ex in rows else 'best'}  " + "  ".join(f"{l}={v/ref:.4f}" for l, v in rows.items()))
    print('  times:', {l: round(v.get('seconds', 0), 2) for l, v in d['selections'].items() if 'seconds' in v})
