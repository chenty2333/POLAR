import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import roots_jacobi, roots_legendre

def poles(a, upper, s, order):
    J = max(0, int(np.ceil(np.log2(upper / a))))
    R, c = a * 2.0**J, np.sin(np.pi * s) / np.pi
    y, w = roots_jacobi(order, 0, -s)
    v = (1 + y) / 2
    tau = list(a * v)
    omega = list(c * a**(1 - s) * w * 2.0**(s - 1))
    y, w = roots_legendre(order)
    t = (3 + y) / 2
    for j in range(J):
        A = a * 2.0**j
        tau.extend(A * t)
        omega.extend(c * A**(1 - s) * w * t**(-s) / 2)
    y, w = roots_jacobi(order, 0, s - 1)
    v = (1 + y) / 2
    tau.extend(R / v)
    omega.extend(c * R**(1 - s) * w * 2.0**(-s) / v)
    return np.array(tau), np.array(omega)

rng = np.random.default_rng(7)
n, sigma, s = 50, 0.5, 0.5
W = np.triu(rng.random((n, n)) < 0.08, 1)
W = (W + W.T).astype(float)
i, j = np.arange(n), (np.arange(n) + 1) % n
W[i, j] = W[j, i] = 1.0
M = np.diag(W.sum(axis=1)) - W + sigma * np.eye(n)
edges = np.argwhere(np.triu(W == 0, 1))
E = np.zeros((n, len(edges)))
E[edges[:, 0], np.arange(len(edges))] = 1
E[edges[:, 1], np.arange(len(edges))] = -1
lam = np.linalg.eigvalsh(M)
base = np.sum(lam**(-s))
exact = np.array([
    np.sum(np.linalg.eigvalsh(M + np.outer(b, b))**(-s)) - base
    for b in E.T
])
assert np.all(exact < 0)
print(f"n={n}, edges={int(W.sum()/2)}, candidates={len(edges)}")
for order in (4, 8, 12, 16):
    tau, omega = poles(sigma, lam[-1] + 2, s, order)
    approx = np.zeros(len(edges))
    for tj, wj in zip(tau, omega):
        X = cho_solve(cho_factor(M + tj * np.eye(n)), E)
        rho, h = np.sum(E * X, axis=0), np.sum(X * X, axis=0)
        approx -= wj * h / (1 + rho)
    rel = np.max(np.abs(approx - exact) / np.abs(exact))
    bound = 256 * 4.0**(-order)
    print(f"r={order:2d}, p={len(tau):3d}, max_rel={rel:.3e}, bound={bound:.3e}")
