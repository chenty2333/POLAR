"""SLQ-Greedy: the trace-difference baseline. Each marginal Tr((M+bb^T)^{-s}) - Tr(M^{-s}) is estimated by
Hutchinson probes (shared across candidates and the base matrix, i.e. common random numbers) with
m-step Lanczos quadrature for z^T f(A) z. All candidates and probes are batched on the GPU."""
import time
import numpy as np
import cupy as cp
import cupyx


def _lanczos_quad(apply, Z, m, f):
    """Batched Lanczos quadrature: returns z_c^T f(A_c) z_c for every column c of Z (n x C)."""
    n, C = Z.shape; nz = cp.sqrt(cp.sum(Z * Z, 0)); V = Z / nz; Vp = cp.zeros_like(V)
    al = cp.zeros((m, C)); be = cp.zeros((m, C)); b = cp.zeros(C)
    for j in range(m):
        W = apply(V) - b * Vp
        a = cp.sum(W * V, 0); W -= a * V
        b = cp.sqrt(cp.sum(W * W, 0)); al[j] = a; be[j] = b
        Vp = V; V = W / cp.where(b > 0, b, 1.0)
    T = cp.zeros((C, m, m)); idx = cp.arange(m)
    T[:, idx, idx] = al.T; T[:, idx[:-1], idx[1:]] = be[:-1].T; T[:, idx[1:], idx[:-1]] = be[:-1].T
    th, U = cp.linalg.eigh(T)
    return nz ** 2 * cp.sum(U[:, 0, :] ** 2 * f(cp.maximum(th, 1e-300)), axis=1)


def slq_greedy(G, cand, k, s, probes=20, m=40, chunk=4000, seed=7):
    f = lambda x: x ** (-s)
    n = G.n; rng = np.random.default_rng(seed); live = np.ones(len(cand), bool); seq = []
    added_a, added_b = [], []
    t0 = time.perf_counter(); spmv = 0

    def base_apply(X):
        Y = G.M0 @ X
        if added_a:
            ia = cp.asarray(added_a); ib = cp.asarray(added_b)
            d = X[ia] - X[ib]
            cupyx.scatter_add(Y, ia, d); cupyx.scatter_add(Y, ib, -d)
        return Y

    for _ in range(k):
        Z = cp.asarray(rng.choice([-1.0, 1.0], size=(n, probes)))
        base = float(cp.mean(_lanczos_quad(base_apply, Z, m, f))); spmv += probes * m
        ids = np.flatnonzero(live); est = np.empty(len(ids))
        per = max(1, chunk // probes)
        for lo in range(0, len(ids), per):
            sub = ids[lo:lo + per]; c = len(sub)
            ea = cp.asarray(np.repeat(cand[sub, 0], probes)); eb = cp.asarray(np.repeat(cand[sub, 1], probes))
            cols = cp.arange(c * probes)

            def apply(X):
                Y = base_apply(X); d = X[ea, cols] - X[eb, cols]
                Y[ea, cols] += d; Y[eb, cols] -= d
                return Y
            q = _lanczos_quad(apply, cp.tile(Z, (1, c)), m, f).reshape(c, probes)
            est[lo:lo + c] = cp.asnumpy(base - cp.mean(q, 1)); spmv += c * probes * m
        e = int(ids[np.argmax(est)]); seq.append(e); live[e] = False
        added_a.append(int(cand[e, 0])); added_b.append(int(cand[e, 1]))
    cp.cuda.Device().synchronize()
    return dict(sequence=seq, seconds=time.perf_counter() - t0, spmv=spmv, probes=probes, lanczos_steps=m)
