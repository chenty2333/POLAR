"""Ratios of every effect-mode selection to the CPU exact greedy references (results/exactcpu_*.json).
Effect gains come from the certified evaluator; exact gains from eigenvalues (they agree to ~1e-11)."""
import sys, json, numpy as np
KEY = {'frac:0.25': 'frac(s=0.25)', 'frac:0.5': 'frac(s=0.5)', 'frac:0.75': 'frac(s=0.75)', 'er:1.0': 'er', 'logdet:0.0': 'logdet'}
REF = {'frac:0.25': 'Exact(s=0.25)', 'frac:0.5': 'Exact(s=0.5)', 'frac:0.75': 'Exact(s=0.75)', 'er:1.0': 'Exact-ER', 'logdet:0.0': 'Exact-Logdet'}
for name in sys.argv[1:]:
    for suf in ('', '_2hop'):
        try:
            c = json.load(open(f'results/exactcpu_{name}{suf}.json')); e = json.load(open(f'results/effect_{name}{suf}.json'))
        except FileNotFoundError:
            continue
        print(f'== {name}{suf or " uniform"}')
        for ti, t in enumerate(c['targets']):
            if KEY[t] not in e['gains'] or REF[t] not in c['runs']:
                continue
            ref = c['runs'][REF[t]]['prefix_gains'][ti][-1]; tab = e['gains'][KEY[t]]
            fin = {l: v[-1] for l, v in tab.items() if not l.startswith('_')}
            rnd = [v for l, v in fin.items() if l.startswith('Random')]
            rows = {l: v for l, v in fin.items() if not l.startswith('Random')}
            if rnd: rows['Random(mean)'] = float(np.mean(rnd))
            for m, r in c['runs'].items():
                if m != REF[t]: rows[m] = r['prefix_gains'][ti][-1]
            print(f'  [{KEY[t]}] vs {REF[t]}: ' + '  '.join(f'{l}={v / ref:.4f}' for l, v in rows.items()))
