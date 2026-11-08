"""GPU (CuPy) implementation of POLAR, its baselines, and a certified true-gain evaluator.

Matrix model: M_S = L_0 + sigma I + sum_{e in S} b_e b_e^T, objective f_s(S) = Tr(M_0^{-s}) - Tr(M_S^{-s}).
All Krylov work is a block multi-shift CG (Jegerlehner recurrences) whose shifted vectors are stored
only on the candidate endpoint rows V_C; full squared norms are tracked by three scalars per pole.
"""
import math, time
import numpy as np
import cupy as cp
import cupyx
import cupyx.scipy.sparse as csp
import scipy.sparse as sp
from spmm import spmm


# ----------------------------------------------------------------------------------------------
# Positive composite Stieltjes quadrature (Section III-A); identical construction to the paper.
def composite(s, a, z, eps):
    r = int(np.ceil(np.log(256 / eps) / np.log(4)))
    K = int(np.ceil(np.log2(z / a)))
    cs = np.sin(np.pi * s) / np.pi

    def rule(beta):
        i = np.arange(r, dtype=float); aa = np.empty(r); aa[0] = beta / (beta + 2); i = i[1:]
        aa[1:] = beta * beta / ((2 * i + beta) * (2 * i + beta + 2))
        bb = 2 / (2 * i + beta) * np.sqrt(i * i * (i + beta) ** 2 / ((2 * i + beta - 1) * (2 * i + beta + 1)))
        x, Q = np.linalg.eigh(np.diag(aa) + np.diag(bb, 1) + np.diag(bb, -1))
        return (1 + x) / 2, Q[0] ** 2 / (beta + 1)

    lo, A = rule(-s); hi, B = rule(s - 1); mid, W = np.polynomial.legendre.leggauss(r)
    ts = [a * lo]; ws = [cs * a ** (1 - s) * A]
    for ell in range(K):
        L = a * 2 ** ell; t = L * (3 + mid) / 2; ts.append(t); ws.append(cs * L / 2 * W * t ** (-s))
    C = a * 2 ** K; ts.append(C / hi); ws.append(cs * C ** (1 - s) * B / hi)
    tau = np.concatenate(ts); om = np.concatenate(ws); o = np.argsort(tau)
    return tau[o], om[o], 256 * 4.0 ** (-r)


def poles(obj, s, sigma, Lam, eps):
    """obj in {'frac','er','logdet'}; 'er' is s=1 (one pole at 0), 'logdet' uses the same single pole."""
    if obj in ('er', 'logdet'):
        return np.zeros(1), np.ones(1), 0.0
    return composite(s, sigma, Lam, eps)


# ----------------------------------------------------------------------------------------------
class Graph:
    def __init__(self, n, edges, sigma, weights=None):
        self.n = int(n); self.sigma = float(sigma)
        u, v = edges[:, 0], edges[:, 1]
        w = np.ones(len(u)) if weights is None else np.asarray(weights, float)
        deg = np.bincount(u, w, n) + np.bincount(v, w, n)
        A = sp.coo_matrix((np.r_[w, w], (np.r_[u, v], np.r_[v, u])), shape=(n, n)).tocsr()
        M = (sp.diags(deg + sigma) - A).tocsr(); M.sort_indices()
        self.M0 = csp.csr_matrix(M)
        self.dmax = float(deg.max()); self.m = len(u)
        self.edges = edges; self.weights = w

    def Lam(self, k):
        return self.sigma + 2 * self.dmax + 4 * k


class Op:
    """y = (M_0 + sum_S b b^T + shift I) x, with the added edges applied as gathers/scatters."""
    def __init__(self, G):
        self.M0 = G.M0; self.A = []; self.B = []; self._ia = None

    def add(self, a, b):
        self.A.append(int(a)); self.B.append(int(b)); self._ia = None

    def __call__(self, x, shift):
        y = spmm(self.M0, x) if x.ndim == 2 else self.M0 @ x
        if shift:
            y += shift * x
        if self.A:
            if self._ia is None:
                self._ia = cp.asarray(self.A); self._ib = cp.asarray(self.B)
            diff = x[self._ia] - x[self._ib]
            cupyx.scatter_add(y, self._ia, diff); cupyx.scatter_add(y, self._ib, -diff)
        return y


class Counter:
    def __init__(self):
        self.spmv = 0; self.rhs = 0


def msolve(op, W, tau, active, tol, maxit=100000, track=False, counter=None):
    """Block multi-shift CG for (M + tau_j I) U = W, all columns of W, all poles at once.
    Returns X[p, |active|, R] = rows of U on `active`, tracked ||U_j[:,c]||^2 (if track), iterations."""
    n, R = W.shape; p = len(tau)
    tau_d = cp.asarray(tau); shift = (tau_d - tau_d[0])[:, None]
    r = W.copy(); d = r.copy(); rr = cp.sum(r * r, axis=0); stop = tol * cp.sqrt(rr)
    act = rr > 0
    X = cp.zeros((p, len(active), R)); P = cp.broadcast_to(W[active][None], (p, len(active), R)).copy()
    z = cp.ones((p, R)); zp = cp.ones((p, R)); live = cp.broadcast_to(act[None], (p, R)).copy()
    if track:
        h = cp.zeros((p, R)); ip = cp.zeros((p, R)); pn = cp.broadcast_to(rr[None], (p, R)).copy()
    ap = cp.ones(R); bp = cp.zeros(R); iters = cp.zeros(R, dtype=cp.int64)
    for it in range(maxit):
        v = op(d, float(tau[0]))
        dv = cp.sum(d * v, axis=0)
        a = cp.where(act, rr / cp.where(act, dv, 1.0), 0.0)
        rn = r - a * v
        rrn = cp.sum(rn * rn, axis=0)
        be = cp.where(act, rrn / cp.where(act, rr, 1.0), 0.0)
        c = cp.where(act, a * bp / ap, 0.0)
        zn = z / (1 + a * shift + c * (1 - z / zp))
        fac = zn / z
        m = live & act[None]
        aj = cp.where(m, a * fac, 0.0); bj = be * fac * fac
        X += aj[:, None, :] * P
        P = cp.where(m[:, None, :], zn[:, None, :] * rn[active][None] + bj[:, None, :] * P, P)
        if track:
            h += 2 * aj * ip + aj * aj * pn
            ip = cp.where(m, bj * (ip + aj * pn), ip)
            pn = cp.where(m, zn * zn * rrn + bj * bj * pn, pn)
        zp = cp.where(m, z, zp); z = cp.where(m, zn, z)
        nr = cp.sqrt(rrn)
        live &= ~(m & (cp.abs(zn) * nr <= stop))
        iters += act
        done = act & (nr <= stop)
        d = cp.where(act, rn + be * d, d); r = cp.where(act, rn, r); rr = cp.where(act, rrn, rr)
        ap = cp.where(act, a, ap); bp = cp.where(act, be, bp)
        act &= ~done
        if it % 8 == 7 and not bool(act.any()):
            break
    else:
        raise RuntimeError('multi-shift CG iteration cap')
    it_h = cp.asnumpy(iters)
    if counter is not None:
        counter.spmv += int(it_h.sum()); counter.rhs += R
    return X, (h if track else None), it_h


# ----------------------------------------------------------------------------------------------
def _block_rows(rows, q, rng, center=False):
    """Independent orthogonal q x q row blocks (block-row orthogonalization of the paper)."""
    nb = -(-rows // q); W = rng.normal(size=(nb, q, q))
    Q, Rm = np.linalg.qr(W)
    Q = Q * np.sign(np.diagonal(Rm, axis1=1, axis2=2))[:, None, :]
    W = np.transpose(Q, (0, 2, 1)).reshape(nb * q, q)[:rows]
    if center:
        W = W - W.mean(0)
    return W


def sketches(G, cand, q, seed, kind='block', chunk=4_000_000):
    """Z (n x q) node sketch, F = C_0^T U (n x q) for base incidence rows, Gc (N x q) candidate rows."""
    rng = np.random.default_rng(seed); n = G.n
    if kind == 'gauss':
        Z = rng.normal(size=(n, q)) / np.sqrt(q)
    else:
        Z = _block_rows(n, q, rng, center=True)
    F = np.zeros((n, q)); u, v = G.edges[:, 0], G.edges[:, 1]; sw = np.sqrt(G.weights)
    for lo in range(0, G.m, chunk):
        hi = min(G.m, lo + chunk)
        U = rng.normal(size=(hi - lo, q)) / np.sqrt(q) if kind == 'gauss' else _block_rows(hi - lo, q, rng)
        U *= sw[lo:hi, None]
        C = sp.coo_matrix((np.r_[np.ones(hi - lo), -np.ones(hi - lo)],
                           (np.r_[u[lo:hi], v[lo:hi]], np.r_[np.arange(hi - lo), np.arange(hi - lo)])),
                          shape=(n, hi - lo)).tocsr()
        F += C @ U
    Gc = rng.normal(size=(len(cand), q)) / np.sqrt(q) if kind == 'gauss' else _block_rows(len(cand), q, rng)
    return Z, F, Gc


# ----------------------------------------------------------------------------------------------
def chunk_width(n, p, nA, budget=10e9):
    """Columns per block solve so that ~10 n-length and ~4 (p x nA) work arrays fit in `budget` bytes."""
    per_col = 8.0 * (10 * n + 4 * p * nA)
    cap = 16 if n > 3e6 else 64        # n > 3e6 (LiveJournal): CG temporaries + pool fragmentation exceed 24 GB at ~29
    return int(max(8, min(cap, budget // per_col)))


def _sync():
    cp.cuda.Device().synchronize()


def polar(G, cand, k, s, q=128, B=16, eps=1e-2, tol=1e-7, obj='frac', sketch=None, naive=False,
          seed=2026, kind='block', trace=None, time_limit=None):
    """POLAR with the cache backend (shared-direction scalar norm updates, exact proxy top-B),
    algebraic shortlist re-ranking and champion reuse. naive=True recomputes both sketch banks and all
    candidate norms every round and selects the proxy maximizer (the Naive Multi-Shift Sketch baseline).
    Returns dict(sequence, seconds, spmv, rhs, p)."""
    n, N = G.n, len(cand); sigma = G.sigma
    tau, om, _ = poles(obj, s, sigma, G.Lam(k), eps); p = len(tau)
    if sketch is None:
        sketch = sketches(G, cand, q, seed, kind)
    Z, F, Gc = sketch
    active_h = np.unique(cand); es_h = np.searchsorted(active_h, cand)
    active = cp.asarray(active_h); ea = cp.asarray(es_h[:, 0]); eb = cp.asarray(es_h[:, 1])
    tau_d = cp.asarray(tau)[:, None]; om_d = cp.asarray(om)[:, None]
    Zd = cp.asarray(Z); Fd = cp.asarray(F); Gd = cp.asarray(Gc)
    op = Op(G); cnt = Counter(); live = np.ones(N, bool); seq = []
    _sync(); t0 = time.perf_counter()

    width = chunk_width(n, p, len(active_h))

    def bank():
        RHS = cp.concatenate([Zd, Fd], axis=1)
        XY = cp.concatenate([msolve(op, RHS[:, lo:lo + width], tau, active, tol, counter=cnt)[0]
                             for lo in range(0, 2 * q, width)], axis=2)
        X, Y = XY[:, :, :q], XY[:, :, q:]
        h = cp.sum((X[:, ea] - X[:, eb]) ** 2, axis=-1); t = cp.sum((Y[:, ea] - Y[:, eb]) ** 2, axis=-1)
        return cp.ascontiguousarray(X), cp.ascontiguousarray(Y), h, t

    def proxy(h, t):
        if obj == 'logdet':
            return cp.log1p(t[0] + sigma * h[0])
        return cp.sum(om_d * h / (1 + t + (sigma + tau_d) * h), axis=0)

    X, Y, h, t = bank()
    for rnd in range(k):
        if rnd and time_limit is not None and time.perf_counter() - t0 > time_limit:
            break
        if naive and rnd:
            X, Y, h, t = bank()
        sc = proxy(h, t); sc[cp.asarray(np.flatnonzero(~live))] = -cp.inf
        Br = 1 if naive else min(B, int(live.sum()))
        short = cp.asnumpy(cp.argsort(-sc)[:Br])
        if naive:
            e = int(short[0])
        else:
            Wb = cp.zeros((n, Br)); a_, b_ = cand[short, 0], cand[short, 1]
            Wb[cp.asarray(a_), cp.arange(Br)] = 1.0; Wb[cp.asarray(b_), cp.arange(Br)] = -1.0
            need_track = obj != 'logdet' and Br > 1
            U, hh, _ = msolve(op, Wb, tau, active, tol, track=need_track, counter=cnt)
            ar = cp.arange(Br)
            bu = U[:, cp.asarray(es_h[short, 0]), ar] - U[:, cp.asarray(es_h[short, 1]), ar]
            if Br == 1:
                j = 0
            elif obj == 'logdet':
                j = int(cp.argmax(cp.log1p(bu[0])))
            else:
                j = int(cp.argmax(cp.sum(om_d * hh / (1 + bu), axis=0)))
            e = int(short[j]); u = U[:, :, j]
        if trace is not None:
            trace.append(dict(shortlist=[int(x) for x in short], chosen=e))
        seq.append(e); live[e] = False
        aa, bb = es_h[e]; a, b = cand[e]
        if not naive and rnd + 1 < k:
            xi = X[:, aa] - X[:, bb]; eta = Gd[e][None] - (Y[:, aa] - Y[:, bb]); D = 1 + u[:, aa] - u[:, bb]
            cc = (u[:, ea] - u[:, eb]) / D[:, None]
            wX = cp.einsum('pnq,pq->pn', X, xi); wY = cp.einsum('pnq,pq->pn', Y, eta)
            h = cp.maximum(0, h - 2 * cc * (wX[:, ea] - wX[:, eb]) + cc * cc * cp.sum(xi * xi, 1)[:, None])
            t = cp.maximum(0, t + 2 * cc * (wY[:, ea] - wY[:, eb]) + cc * cc * cp.sum(eta * eta, 1)[:, None])
            X -= u[:, :, None] * xi[:, None, :] / D[:, None, None]
            Y += u[:, :, None] * eta[:, None, :] / D[:, None, None]
        Fd[a] += Gd[e]; Fd[b] -= Gd[e]
        op.add(a, b)
    _sync()
    return dict(sequence=seq, seconds=time.perf_counter() - t0, spmv=cnt.spmv, rhs=cnt.rhs, p=p,
                rounds=len(seq), truncated=len(seq) < k)


# ----------------------------------------------------------------------------------------------
def degree_greedy(G, cand, k):
    deg = np.bincount(G.edges[:, 0], G.weights, G.n) + np.bincount(G.edges[:, 1], G.weights, G.n)
    live = np.ones(len(cand), bool); seq = []
    for _ in range(k):
        val = np.where(live, deg[cand[:, 0]] + deg[cand[:, 1]], np.inf)
        e = int(np.argmin(val)); seq.append(e); live[e] = False
        deg[cand[e, 0]] += 1; deg[cand[e, 1]] += 1
    return seq


def random_set(cand, k, seed):
    return np.random.default_rng(seed).choice(len(cand), k, replace=False).tolist()


# ----------------------------------------------------------------------------------------------
def true_marginals(G, cand, seq, s, obj='frac', eps=1e-9, tol=1e-10, kmax=None):
    """Certified marginals along a selected sequence. 'frac': composite quadrature with derivative
    relative error <= eps (Theorem 1) and single-RHS multi-shift CG; 'er': exact rational formula
    (one pole at 0); 'logdet': log(1 + b^T M^{-1} b). Returns (marginals, residual bound rho_max, spmv)."""
    kmax = kmax or len(seq)
    tau, om, qerr = poles(obj, s, G.sigma, G.Lam(kmax), eps)
    om_d = cp.asarray(om); op = Op(G); cnt = Counter(); out = []
    for e in seq:
        a, b = int(cand[e, 0]), int(cand[e, 1])
        w = cp.zeros((G.n, 1)); w[a, 0] = 1.0; w[b, 0] = -1.0
        U, hh, _ = msolve(op, w, tau, cp.asarray([a, b]), tol, track=obj != 'logdet', counter=cnt)
        bu = U[:, 0, 0] - U[:, 1, 0]
        if obj == 'logdet':
            g = float(cp.log1p(bu[0]))
        else:
            g = float(cp.sum(om_d * hh[:, 0] / (1 + bu)))
        out.append(g); op.add(a, b)
    rho = tol * float(np.max((G.Lam(kmax) + tau) / (G.sigma + tau)))
    return np.array(out), dict(quadrature_bound=qerr, residual_rho=rho, p=len(tau), spmv=cnt.spmv)
