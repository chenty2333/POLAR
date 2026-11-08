"""Summarize results/time_*.json: median seconds, spmv, rhs per method, speedup vs Naive."""
import sys, json, numpy as np
for p in sys.argv[1:]:
    d = json.load(open(p)); g = d['graph']; R = d['runs']
    nv = np.median([r['seconds'] for r in R['Naive']]) if 'Naive' in R else None
    print(f"== {g['name']} n={g['n']} m={g['m']} active={g.get('n_active')} p={next(iter(R.values()))[0].get('p')}")
    for m, rs in R.items():
        t = np.median([r['seconds'] for r in rs]); L = len(R['Naive'][0]['sequence']) if 'Naive' in R else 0; same = all(r['sequence'][:L] == R['Naive'][0]['sequence'] for r in rs) if 'Naive' in R else None
        tr = any(r.get('truncated') for r in rs); rd = rs[0].get('rounds')
        print(f"   {m:14s} {'>' if tr else ' '}{t:9.2f}s" + (f" (stopped after {rd} rounds)" if tr else '') + f"  spmv={rs[0].get('spmv')}  rhs={rs[0].get('rhs')}" + (f"  x{nv / t:.1f}" if nv else '') + (f"  same-as-naive={same}" if same is not None else ''))
