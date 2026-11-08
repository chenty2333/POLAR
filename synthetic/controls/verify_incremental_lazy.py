import heapq, os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
from time import perf_counter
import numpy as np
from scipy.linalg import solve
from scipy.special import roots_jacobi
rng = np.random.default_rng(7)
n, N, k, q, p, s, sigma = 140, 2000, 15, 24, 24, 0.5, 1.0
W = np.triu(rng.random((n, n)) < 0.035, 1)
W = (W + W.T).astype(float)
i = np.arange(n)
W[i, (i + 1) % n] = W[(i + 1) % n, i] = 1
M0 = np.diag(W.sum(1)) - W + sigma * np.eye(n)
E = np.argwhere(np.triu(W == 0, 1))
u, v = E[rng.choice(len(E), N, replace=False)].T
y, w = roots_jacobi(p, -s, s - 1)
tau = (1 - y) / (1 + y)
omega = 2 * np.sin(np.pi * s) / np.pi * w / (1 + y)
def initial(M):
    R = np.array([solve(M + t * np.eye(n), np.eye(n), assume_a='pos') for t in tau])
    return R @ Z, R[:, u, u] + R[:, v, v] - 2 * R[:, u, v]
def norms(X, ids):
    return np.linalg.norm(X[:, u[ids]] - X[:, v[ids]], axis=-1)
def score(h, d):
    return np.sum(omega[:, None] * h * h / (1 + d), axis=0)
def run(lazy, audit=False):
    start = perf_counter()
    M, active, seq, count, err = M0.copy(), np.ones(N, bool), [], 0, 0.0
    if lazy:
        X, d = initial(M)
        H = norms(X, np.arange(N))
        stamp, count = np.zeros(N, int), N
    for r in range(k):
        ids = np.flatnonzero(active)
        if not lazy:
            X, d = initial(M)
            values = score(norms(X, ids), d[:, ids])
            refs.append((ids.copy(), values.copy()))
            e = ids[np.argmax(values)]
            count += len(ids)
        else:
            if audit:
                ids0, exact = refs[r]
                values = score(norms(X, ids0), d[:, ids0])
                err = max(err, np.max(np.abs(values - exact) / np.abs(exact)))
            upper = score(H[:, ids], d[:, ids]) * (1 + 1e-12)
            heap = [(-a, int(e)) for a, e in zip(upper, ids)]
            heapq.heapify(heap)
            while True:
                _, e = heapq.heappop(heap)
                if stamp[e] == r:
                    break
                H[:, e:e+1] = norms(X, np.array([e]))
                value = score(H[:, e:e+1], d[:, e:e+1])[0]
                stamp[e] = r
                count += 1
                heapq.heappush(heap, (-value, e))
        seq.append(e)
        active[e] = False
        b = np.zeros(n)
        b[u[e]], b[v[e]] = 1, -1
        if lazy and r + 1 < k:
            U = np.array([solve(M + t * np.eye(n), b, assume_a='pos') for t in tau])
            c, V = U[:, u] - U[:, v], X[:, u[e]] - X[:, v[e]]
            den = 1 + U[:, u[e]] - U[:, v[e]]
            H += np.abs(c) * np.linalg.norm(V, axis=1)[:, None] / den[:, None]
            X -= U[:, :, None] * V[:, None, :] / den[:, None, None]
            d -= c * c / den[:, None]
        M += np.outer(b, b)
    return seq, count, perf_counter() - start, err
for q in (24, n):
    Z = np.eye(n) if q == n else rng.normal(size=(n, q)) / np.sqrt(q)
    refs = []
    a, ca, ta, _ = run(False)
    b, cb, tb, _ = run(True)
    _, _, _, err = run(True, audit=True)
    assert a == b and err < 1e-10
    print(f'q={q}, max_relative_error={err:.3e}, exact_match={a == b}')
    print(f'evaluations={ca}/{cb}, pruning={100*(1-cb/ca):.2f}%')
    print(f'baseline={ta:.6f}s, incremental_lazy={tb:.6f}s, speedup={ta/tb:.3f}x')
