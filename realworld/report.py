"""Build a Markdown summary of all real-network results in results/ (run locally after rsync)."""
import json, glob, os
from pathlib import Path
import numpy as np

R = Path(__file__).parent / 'results'
out = []
P = out.append


def load(name):
    p = R / name
    return json.loads(p.read_text()) if p.exists() else None


def fin(tab):
    f = {l: v[-1] for l, v in tab.items() if not l.startswith('_')}
    rnd = [v for l, v in f.items() if l.startswith('Random')]
    f = {l: v for l, v in f.items() if not l.startswith('Random')}
    if rnd:
        f['Random'] = float(np.mean(rnd))
    return f


EXK = {'frac(s=0.25)': 'Exact(s=0.25)', 'frac(s=0.5)': 'Exact(s=0.5)', 'frac(s=0.75)': 'Exact(s=0.75)', 'er': 'Exact-ER', 'logdet': 'Exact-Logdet'}
POK = {'frac(s=0.25)': 'POLAR(s=0.25)', 'frac(s=0.5)': 'POLAR(s=0.5)', 'frac(s=0.75)': 'POLAR(s=0.75)', 'er': 'Sketched-ER', 'logdet': 'Sketched-Logdet'}
SLQK = {'frac(s=0.25)': 'SLQ(s=0.25)', 'frac(s=0.5)': 'SLQ(s=0.5)', 'frac(s=0.75)': 'SLQ(s=0.75)'}

# ---------------------------------------------------------------- datasets
P('## Datasets\n\n| Graph | n | m | d_max |\n|---|---:|---:|---:|')
seen = set()
for f in sorted(R.glob('*.json'), key=lambda p: (json.loads(p.read_text()).get('graph', {}).get('m', 0) if 'graph' in json.loads(p.read_text()) and isinstance(json.loads(p.read_text())['graph'], dict) else 0)):
    d = json.loads(f.read_text()); g = d.get('graph')
    if not isinstance(g, dict) or 'n' not in g or g['name'] in seen:
        continue
    seen.add(g['name']); P(f"| {g['name']} | {g['n']:,} | {g['m']:,} | {int(g.get('dmax', 0))} |")

# ---------------------------------------------------------------- optimum
P('\n## Tiny graphs: ratio to the brute-force optimum (N=40 candidates)\n')
P('| Graph | s | k | POLAR | Exact greedy | Exact-ER | Exact-Logdet | Degree | Random |\n|---|---|---|---|---|---|---|---|---|')
for f in sorted(R.glob('opt_*.json')):
    d = json.loads(f.read_text())
    for r in d['rows']:
        m = r['methods']; o = r['opt']; rnd = np.mean([v for l, v in m.items() if l.startswith('Random')])
        P(f"| {d['graph']['name']} | {d['s']} | {r['k']} | {m['POLAR'] / o:.4f} | {m['Exact'] / o:.4f} | {m['Exact-ER'] / o:.4f} | "
          f"{m['Exact-Logdet'] / o:.4f} | {m['Degree'] / o:.4f} | {rnd / o:.3f} |")

# ---------------------------------------------------------------- quality vs exact
P('\n## Quality relative to exact greedy on the same target (k=20, N=2000)\n')
P('Each cell: method gain / exact-greedy gain for the target in the column header. CPU exact references for ca-HepPh / ca-CondMat.\n')


def exact_ref(name, suf):
    c = load(f'exactcpu_{name}{suf}.json')
    if not c:
        return {}
    T = {'frac:0.25': 'frac(s=0.25)', 'frac:0.5': 'frac(s=0.5)', 'frac:0.75': 'frac(s=0.75)', 'er:1.0': 'er', 'logdet:0.0': 'logdet'}
    return {lab: {T[t]: r['prefix_gains'][i][-1] for i, t in enumerate(c['targets'])} for lab, r in c['runs'].items()}, c


for suf, pool in (('', 'uniform'), ('_2hop', '2-hop')):
    P(f'\n### {pool} pool\n')
    P('| Graph | target | POLAR(same s) | SLQ(same s) | Exact, other exponents (min) | Sketched-ER | Degree | Random | POLAR s | Exact s |')
    P('|---|---|---|---|---|---|---|---|---|---|')
    for name in ('email-Eu-core', 'Hamster', 'ca-GrQc', 'ca-HepPh', 'ca-CondMat'):
        e = load(f'effect_{name}{suf}.json')
        if not e:
            continue
        cx = exact_ref(name, suf); cref = cx[0] if cx else {}
        for key in ('frac(s=0.25)', 'frac(s=0.5)', 'frac(s=0.75)', 'er', 'logdet'):
            if key not in e['gains']:
                continue
            f = fin(e['gains'][key]); exl = EXK[key]
            ref = f.get(exl) or cref.get(exl, {}).get(key)
            if ref is None:
                continue
            others = [v for l, v in f.items() if l.startswith('Exact') and l != exl] + \
                     [g[key] for l, g in cref.items() if l != exl and key in g]
            slq = f.get(SLQK.get(key, ''), None)
            tp = e['selections'].get(POK[key], {}).get('seconds')
            te = e['selections'].get(exl, {}).get('seconds') or (cx[1]['runs'].get(exl, {}).get('seconds') if cx else None)
            P(f"| {name} | {key} | {f[POK[key]] / ref:.4f} | {'' if slq is None else f'{slq / ref:.4f}'} | "
              f"{(min(others) / ref if others else float('nan')):.4f} | {f['Sketched-ER'] / ref:.4f} | {f['Degree'] / ref:.4f} | {f.get('Random', float('nan')) / ref:.3f} | "
              f"{'' if tp is None else f'{tp:.1f}'} | {'' if te is None else f'{te:.0f}' + (' (CPU)' if cx and exl in cx[1]['runs'] and exl not in e['selections'] else '')} |")

# ---------------------------------------------------------------- large graphs, best-relative
P('\n## Large graphs: ratio to the best method on each target (no exact reference)\n')
P('| Graph | target | POLAR(same s) | best other POLAR | Sketched-ER | Sketched-Logdet | Degree | Random |\n|---|---|---|---|---|---|---|---|')
for name in ('email-Enron', 'com-DBLP', 'com-Amazon', 'web-Google', 'roadNet-PA', 'com-Youtube', 'roadNet-CA', 'as-Skitter', 'com-LiveJournal'):
    e = load(f'effect_{name}.json')
    if not e:
        continue
    for key in [k for k in e['gains'] if k.startswith('frac')]:
        f = fin(e['gains'][key])
        if POK[key] not in f:
            continue
        best = max(f.values()); op = [v for l, v in f.items() if l.startswith('POLAR') and l != POK[key]]
        P(f"| {name} | {key} | {f[POK[key]] / best:.4f} | {'' if not op else f'{max(op) / best:.4f}'} | {f['Sketched-ER'] / best:.4f} | "
          f"{f['Sketched-Logdet'] / best:.4f} | {f['Degree'] / best:.4f} | {f.get('Random', float('nan')) / best:.3f} |")

# ---------------------------------------------------------------- timing
P('\n## Running time, s=0.5, k=20, q=128 (RTX 4090; median over repetitions; Naive once, capped)\n')
P('| Graph | n | m | Naive (s) | POLAR B=1 (s) | speedup | POLAR B=16 (s) | speedup | B=1 = Naive? | Naive s/round |\n|---|---:|---:|---:|---:|---:|---:|---:|---|---:|')
for f in sorted(R.glob('time_*.json'), key=lambda p: json.loads(p.read_text())['graph']['m']):
    d = json.loads(f.read_text()); g = d['graph']; ru = d['runs']
    nv = ru['Naive'][0]; tn = nv['seconds']; tr = nv.get('truncated', False); rd = nv.get('rounds', len(nv['sequence']))
    t1 = float(np.median([r['seconds'] for r in ru['POLAR(B=1)']])); t16 = float(np.median([r['seconds'] for r in ru['POLAR(B=16)']]))
    same = all(r['sequence'][:len(nv['sequence'])] == nv['sequence'] for r in ru['POLAR(B=1)'])
    gt = '>' if tr else ''
    P(f"| {g['name']} | {g['n']:,} | {g['m']:,} | {gt}{tn:.1f}{f' ({rd}/20 rounds)' if tr else ''} | {t1:.1f} | {gt}{tn / t1:.1f}x | {t16:.1f} | {gt}{tn / t16:.1f}x | "
      f"{'yes' if same else 'NO'} | {tn / rd:.1f} |")

# ---------------------------------------------------------------- parameters
d = load('param_ca-GrQc.json')
if d:
    P('\n## Parameter study, ca-GrQc, s=0.5 (single sketch seed): gain / exact greedy\n')
    P('| q | B | eps | ratio | seconds | poles p |\n|---|---|---|---|---|---|')
    for r in d['rows']:
        P(f"| {r['q']} | {r['B']} | {r['eps']} | {r['gain'] / r['ref_gain']:.4f} | {r['seconds']:.2f} | {r['p']} |")

P('\n## Sketch grid (5 seeds): mean / min ratio to exact greedy, s=0.5\n')
P('| Graph | kind | q | B=4 | B=16 | B=32 |\n|---|---|---|---|---|---|')
for name in ('email-Eu-core', 'Hamster', 'ca-GrQc'):
    d = load(f'diag_grid_{name}.json')
    if not d:
        continue
    for kind in ('block', 'gauss'):
        for q in (24, 64, 128):
            cells = []
            for B in (4, 16, 32):
                a = np.array([r['ratio'] for r in d['rows'] if r['kind'] == kind and r['q'] == q and r['B'] == B])
                cells.append(f'{a.mean():.4f} / {a.min():.4f}')
            P(f'| {name} | {kind} | {q} | ' + ' | '.join(cells) + ' |')

P('\n## Per-round marginal recovery r_t = g(chosen)/max g along POLAR trajectories (5 seeds, s=0.5)\n')
P('| Graph | config | mean r_t (per trajectory) | min r_t (per trajectory) | rounds with r_t=1 |\n|---|---|---|---|---|')
for name in ('email-Eu-core', 'Hamster', 'ca-GrQc'):
    d = load(f'diag_round_{name}.json')
    if not d:
        continue
    cfgs = sorted({(r['kind'], r['q'], r['B']) for r in d['rows']}, key=lambda c: (c[0] != 'block', c[1]))
    for c in cfgs:
        a = np.array([r['r'] for r in d['rows'] if (r['kind'], r['q'], r['B']) == c])
        P(f"| {name} | {c[0]} q={c[1]} B={c[2]} | {a.mean(1).min():.3f}-{a.mean(1).max():.3f} | {a.min(1).min():.3f}-{a.min(1).max():.3f} | {(a > 1 - 1e-9).mean():.0%} |")

P('\n## sigma sweep: cross-evaluation of exact greedy, ratio to best per target (targets f_0.25, f_0.5, f_0.75, ER, logdet)\n')
for name in ('email-Eu-core', 'Hamster', 'ca-GrQc'):
    d = load(f'diag_sigma_{name}.json')
    if not d:
        continue
    for r in d['rows']:
        F = r['final_gain']; best = np.max(np.array(list(F.values())), 0)
        P(f"\n**{name}, sigma={r['sigma']}**\n\n| selection | f_0.25 | f_0.5 | f_0.75 | ER | logdet | Jaccard vs Exact-ER |\n|---|---|---|---|---|---|---|")
        for m, v in F.items():
            P(f'| {m} | ' + ' | '.join(f'{x:.4f}' for x in np.array(v) / best) + f" | {r['jaccard'][m]:.2f} |")

# ---------------------------------------------------------------- downstream
TS = [0.03, 0.1, 0.32, 1, 3.16, 10, 31.6]
P('\n## Heat-trace reduction dH(t) = Tr exp(-tL0) - Tr exp(-tL_S): ratio to the best selection at each t\n')
for f in sorted(R.glob('downstream_*.json')):
    d = json.loads(f.read_text()); t = np.array(d['t']); rows = d['rows']
    lab = [l for l in rows if not l.startswith('Random')]; H = np.array([rows[l]['heat_reduction'] for l in lab]); best = H.max(0)
    idx = [int(np.argmin(abs(t - x))) for x in TS]
    P(f"\n**{d['graph']['name']} ({d['graph'].get('pool', 'uniform')} pool)**\n\n| selection | " + ' | '.join(f't={t[i]:.2g}' for i in idx) + ' |\n|---|' + '---|' * len(idx))
    for l in lab:
        if l.startswith('Sketched-Logdet') or l.startswith('SLQ(s=0.25)') or l.startswith('SLQ(s=0.75)'):
            continue
        P(f'| {l} | ' + ' | '.join(f'{H[lab.index(l), i] / best[i]:.4f}' for i in idx) + ' |')

# ---------------------------------------------------------------- k=100
e = load('effect_ca-GrQc_2hop_k100.json')
if e:
    P('\n## ca-GrQc 2-hop pool, k=100: gain / exact greedy\n')
    P('| target | ' + ' | '.join(l for l in fin(e['gains']['frac(s=0.5)'])) + ' |')
    P('|---|' + '---|' * len(fin(e['gains']['frac(s=0.5)'])))
    for key, tab in e['gains'].items():
        f = fin(tab); ref = f[EXK[key]]
        P(f'| {key} | ' + ' | '.join(f'{v / ref:.4f}' for v in f.values()) + ' |')


# ---------------------------------------------------------------- pool robustness
P('\n## Candidate-pool robustness (uniform pools, seeds ps0/ps1/ps2): POLAR(s) / Sketched-ER / Exact-ER, ratio to Exact(s)\n')
P('| Graph | pool | f_0.25 | f_0.5 | f_0.75 |\n|---|---|---|---|---|')
for g in ('email-Eu-core', 'Hamster', 'ca-GrQc'):
    for ps in ('', '_ps1', '_ps2'):
        e = load(f'effect_{g}{ps}.json')
        if not e:
            continue
        cells = []
        for sv in (0.25, 0.5, 0.75):
            t = e['gains'][f'frac(s={sv})']; ex = t[f'Exact(s={sv})'][-1]
            cells.append(f"{t[f'POLAR(s={sv})'][-1] / ex:.4f} / {t['Sketched-ER'][-1] / ex:.4f} / {t['Exact-ER'][-1] / ex:.4f}")
        P(f"| {g} | {ps.strip('_') or 'ps0'} | " + ' | '.join(cells) + ' |')

# ---------------------------------------------------------------- youtube diag
d = load('ytdiag_com-Youtube.json')
if d:
    best = max(d['ref'].values())
    P('\n## com-Youtube, f_0.5: POLAR configurations vs the best method (Sketched-ER)\n')
    P('| config | gain / best | seconds (concurrency-free) |\n|---|---|---|')
    P(f"| q=128 B=16 (default) | {d['ref']['POLAR(s=0.5)'] / best:.4f} | |")
    for r in d['rows']:
        P(f"| q={r['q']} B={r['B']} | {r['gain'] / best:.4f} | {r['seconds']:.0f} |")

(Path(__file__).parent / 'REPORT_realworld.md').write_text('# POLAR real-network results\n\n' + '\n'.join(out) + '\n')
print('\n'.join(out))
