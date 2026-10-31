"""T8: dense random SPD matrices, truncation to R = 8 (Table VI of the paper).

A = Q Q^T + N I with Q_ij ~ U[0,1), b_i ~ U[0,1) (dense b, so supp(b) = N: the rule "supp(b) is always
included" cannot be applied here and configurations are selected by the criterion of each column only;
the greedy baseline starts from the single configuration argmax|b|).

Columns (fidelity F of the Galerkin solution on R = 8 configurations):
  exact_paper_grid : the 8 configurations with the largest EXACT p_agg on the grid t_j = 0.25 j, j = 1..8
                     (the grid of the 4x4 example; it violates Eq. (nyquist) since dt*dlam/pi >> 1)
  exact_nyquist    : same, on the Nyquist grid t_j = j*pi/dlam, j = 1..8
  random, oracle (largest |x_z|), greedy (classical residual selection).
Mean and standard deviation over NSEEDS independent instances.  Also prints (i) the single instance of the
earlier version of the paper (np.random.seed(N), i.e. problems.random_spd) and (ii) the Lie-Trotter
infidelity with S = 4, t = 2 for the matrices of Sec. "Circuit cost and dense matrices".
"""
import json
import numpy as np
import scipy.sparse as sp
from common import *                                   # noqa: F401,F403
from sqd_linsolve.baselines import greedy_residual     # noqa: F401

R = 8
NSEEDS = 30
NS = [16, 32, 64, 128]


def Fof(A, b, x, C):
    return metrics(A, x, galerkin(A, b, C))["F"]


def pagg(A, b, times):
    return np.mean(exact_probs(A, b, times), axis=0)


def one_instance(A, b, rng):
    x = np.linalg.solve(A, b)
    w = np.linalg.eigvalsh(A)
    dl = float(w[-1] - w[0])
    P1 = pagg(A, b, 0.25 * np.arange(1, 9))
    P2 = pagg(A, b, (np.pi / dl) * np.arange(1, 9))
    r = dict(dl=dl, grid=0.25 * dl / np.pi, t_nyq_max=8 * np.pi / dl,
             exact_paper_grid=Fof(A, b, x, np.argsort(-P1)[:R]),
             exact_nyquist=Fof(A, b, x, np.argsort(-P2)[:R]),
             random=float(np.mean([Fof(A, b, x, rng.choice(len(b), R, replace=False)) for _ in range(200)])),
             oracle=Fof(A, b, x, np.argsort(-np.abs(x))[:R]),
             greedy=metrics(A, x, greedy_residual(A, b, R, [int(np.argmax(np.abs(b)))])[0])["F"])
    return r


out = dict(R=R, NSEEDS=NSEEDS, ensemble={}, single_instance_seed_N={}, trotter={})
for N in NS:
    rows = []
    for s in range(NSEEDS):
        g = np.random.default_rng(1000 * N + s)
        Q = g.random((N, N)); A = Q @ Q.T + N * np.eye(N); b = g.random(N)
        rows.append(one_instance(A, b, g))
    keys = rows[0].keys()
    out["ensemble"][N] = {k: dict(mean=float(np.mean([r[k] for r in rows])), std=float(np.std([r[k] for r in rows], ddof=1)),
                                  min=float(np.min([r[k] for r in rows])), max=float(np.max([r[k] for r in rows]))) for k in keys}
    out["ensemble"][N]["L"] = N * (N + 1) // 2
    # single instance used in the earlier (single-seed) version of the table
    A, b = random_spd(N)
    out["single_instance_seed_N"][N] = one_instance(A, b, np.random.default_rng(N))
    print(N, {k: round(v["mean"], 3) for k, v in out["ensemble"][N].items() if isinstance(v, dict)})
    print("   single instance (seed=N):", {k: round(v, 3) for k, v in out["single_instance_seed_N"][N].items()})

# Lie-Trotter infidelity, S = 4, t = 2, on the single instances N = 16, 32
for N in (16, 32):
    A, b = random_spd(N)
    op = pauli_op(A)
    mats = [(float(c.real), sp.csr_matrix(p.to_matrix(sparse=True))) for p, c in zip(op.paulis, op.coeffs)]
    psi0 = b / np.linalg.norm(b)
    for t in (0.25, 1.0, 2.0):
        ex = sl.expm(-1j * A * t) @ psi0
        psi = psi0.astype(complex)
        for _ in range(4):
            for c, P in mats:
                th = c * t / 4
                psi = np.cos(th) * psi - 1j * np.sin(th) * (P @ psi)
        out["trotter"][f"N{N}_t{t}"] = float(1 - abs(np.vdot(ex, psi)) ** 2)
    print("Trotter infidelity S=4:", {k: round(v, 3) for k, v in out["trotter"].items() if k.startswith(f"N{N}")})
json.dump(out, open("t8.json", "w"), indent=1)
