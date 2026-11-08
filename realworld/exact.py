"""Dense references on the GPU: exact greedy (fractional / ER / log-det), set evaluation, brute force."""
import itertools, time
import numpy as np
import cupy as cp
from polar_gpu import composite


def _add(M, a, b):
    M[a, a] += 1; M[b, b] += 1; M[a, b] -= 1; M[b, a] -= 1


def frac_value(lam, s, obj):
    if obj == 'frac':
        return cp.sum(lam ** (-s))
    if obj == 'er':
        return cp.sum(1 / lam)
    return -cp.sum(cp.log(lam))          # log-det gain is the decrease of -log det


def twin_gain(M, pairs, s):
    """Exact marginal for candidates joining non-adjacent twins (identical neighbour sets) of the current
    unweighted graph: b = e_u - e_v satisfies M b = (d + sigma) b and adding uv moves only that eigenvalue
    to d + sigma + 2, so g = x^{-s} - (x + 2)^{-s} with x = M_uu. Returns nan for non-twin pairs."""
    a = cp.asarray(pairs[:, 0]); b = cp.asarray(pairs[:, 1]); r = cp.arange(len(pairs))
    D = M[a] - M[b]; D[r, a] = 0; D[r, b] = 0
    twin = (cp.abs(D).max(1) == 0) & (M[a, b] == 0) & (M[a, a] == M[b, b])
    x = M[a, a]; g = -x ** (-s) * cp.expm1(-s * cp.log1p(2 / x))
    return cp.asnumpy(cp.where(twin, g, cp.nan))


def exact_greedy(M0, cand, k, s, obj='frac'):
    """True-marginal greedy. 'frac' screens with the eps=1e-9 composite rule of Theorem 1 and resolves
    every interval-surviving contender with eigenvalues (as in the paper's reference)."""
    M = cp.asarray(M0); n = M.shape[0]; live = np.ones(len(cand), bool); seq = []; contenders = []
    ea = cp.asarray(cand[:, 0]); eb = cp.asarray(cand[:, 1]); t0 = time.perf_counter()
    for _ in range(k):
        lam, V = cp.linalg.eigh(M); Wc = (V[ea] - V[eb]) ** 2
        mask = cp.asarray(~live)
        if obj == 'er':
            sc = (Wc @ lam ** -2) / (1 + Wc @ (1 / lam)); sc[mask] = -cp.inf; e = int(cp.argmax(sc))
        elif obj == 'logdet':
            sc = cp.log1p(Wc @ (1 / lam)); sc[mask] = -cp.inf; e = int(cp.argmax(sc))
        else:
            tau, om, eps = composite(s, float(lam[0]) * 0.999, float(lam[-1]) + 2.001, 1e-9)
            eps = max(eps, 1e-9)
            D = 1 / (lam[:, None] + cp.asarray(tau)[None]); sc = ((Wc @ (D * D)) / (1 + Wc @ D)) @ cp.asarray(om)
            sc[mask] = -cp.inf; best_lo = float(cp.max(sc)) / (1 + eps)
            ids = np.flatnonzero(cp.asnumpy(sc / (1 - eps) >= best_lo) & live)
            if len(ids) == 1:
                e = int(ids[0]); contenders.append((1, 0))
            else:
                vals = np.empty(len(ids)); tw = twin_gain(M, cand[ids], s); solved = 0
                old = float(cp.sum(lam ** -s))
                for i, c in enumerate(ids):
                    if np.isfinite(tw[i]):
                        vals[i] = tw[i]; continue
                    A = M.copy(); _add(A, int(cand[c, 0]), int(cand[c, 1])); solved += 1
                    vals[i] = old - float(cp.sum(cp.linalg.eigvalsh(A) ** -s))
                e = int(ids[int(np.argmax(vals))]); contenders.append((len(ids), solved))
        seq.append(e); live[e] = False; _add(M, int(cand[e, 0]), int(cand[e, 1]))
    cp.cuda.Device().synchronize()
    return dict(sequence=seq, seconds=time.perf_counter() - t0, contenders=contenders)


def eval_prefix(M0, cand, seq, s, obj='frac'):
    """Exact gains f(S_r) for every prefix r, from eigenvalues."""
    M = cp.asarray(M0); base = float(frac_value(cp.linalg.eigvalsh(M), s, obj)); out = []
    for e in seq:
        _add(M, int(cand[e, 0]), int(cand[e, 1]))
        out.append(base - float(frac_value(cp.linalg.eigvalsh(M), s, obj)))
    return np.array(out)


def eval_prefix_multi(M0, cand, seq, targets):
    """Prefix gains for several (obj, s) targets from one eigenvalue solve per prefix."""
    M = cp.asarray(M0); lam = cp.linalg.eigvalsh(M)
    base = [float(frac_value(lam, s, o)) for o, s in targets]; out = [[] for _ in targets]
    for e in seq:
        _add(M, int(cand[e, 0]), int(cand[e, 1])); lam = cp.linalg.eigvalsh(M)
        for i, (o, s) in enumerate(targets):
            out[i].append(base[i] - float(frac_value(lam, s, o)))
    return [np.array(v) for v in out]


def brute_force(M0, cand, k, s, obj='frac', batch=4096):
    """Optimal budget-k set by exhaustive enumeration with batched eigenvalue solves."""
    M = cp.asarray(M0); n = M.shape[0]; base = float(frac_value(cp.linalg.eigvalsh(M), s, obj))
    best = -np.inf; arg = None; combos = itertools.combinations(range(len(cand)), k)
    idx = np.arange(n)
    while True:
        chunk = list(itertools.islice(combos, batch))
        if not chunk:
            break
        C = np.asarray(chunk); A = cp.broadcast_to(M, (len(C), n, n)).copy()
        for j in range(k):
            a = cp.asarray(cand[C[:, j], 0]); b = cp.asarray(cand[C[:, j], 1]); r = cp.arange(len(C))
            A[r, a, a] += 1; A[r, b, b] += 1; A[r, a, b] -= 1; A[r, b, a] -= 1
        lam = cp.linalg.eigvalsh(A)
        if obj == 'frac':
            g = base - cp.sum(lam ** (-s), axis=1)
        elif obj == 'er':
            g = base - cp.sum(1 / lam, axis=1)
        else:
            g = base + cp.sum(cp.log(lam), axis=1)
        i = int(cp.argmax(g)); v = float(g[i])
        if v > best:
            best = v; arg = C[i].tolist()
    return best, arg
