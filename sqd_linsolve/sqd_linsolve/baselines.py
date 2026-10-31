"""
MIT License

Copyright (c) 2026 Taha Selim

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


"""Classical reference strategies and exact-propagator sampling emulation (Sec. "Results")."""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from .core import galerkin

__all__ = ["exact_probs", "sample_counts", "krylov_support", "greedy_residual",
           "oracle_set", "random_set"]


def exact_probs(A, b, times):
    """Exact p(z|t_j) = |<z|exp(-iAt_j)|b>|^2 (idealized, Trotter-error-free sampling)."""
    A = sp.csc_matrix(A, dtype=complex)
    psi = np.asarray(b, dtype=complex) / np.linalg.norm(b)
    return [np.abs(spla.expm_multiply(-1j * A * t, psi)) ** 2 for t in times]


def sample_counts(probs_list, shots, rng, readout_flip=0.0, n=None):
    """Multinomial sampling of each p(z|t_j); optional independent bit-flip readout noise."""
    tot = np.zeros(len(probs_list[0]), dtype=np.int64)
    for p in probs_list:
        p = np.clip(p, 0, None); p = p / p.sum()
        idx = rng.choice(len(p), size=shots, p=p)
        if readout_flip > 0:
            flips = np.zeros(shots, dtype=np.int64)
            for q in range(n):
                flips |= (rng.random(shots) < readout_flip).astype(np.int64) << q
            idx = idx ^ flips
        tot += np.bincount(idx, minlength=len(p))
    return tot


def krylov_support(A, b, k):
    """supp(b) U supp(Ab) U ... U supp(A^k b): k sparse matrix-vector products."""
    A = sp.csr_matrix(A); v = np.asarray(b, dtype=float); S = set()
    for _ in range(k + 1):
        S |= set(np.flatnonzero(np.abs(v) > 1e-14).tolist()); v = A @ v
    return np.array(sorted(S))


def greedy_residual(A, b, R, seed=None):
    """Classical selected-CI-style selection: start from supp(b) (or `seed`) and repeatedly
    add the configuration with the largest |(b - A x_approx)_z|."""
    A = sp.csr_matrix(A)
    S = set(np.flatnonzero(np.abs(b) > 1e-14).tolist()) if seed is None else set(seed)
    while len(S) < R:
        x = galerkin(A, b, S); r = np.abs(b - A @ x); r[list(S)] = 0
        S.add(int(np.argmax(r)))
    return galerkin(A, b, S), np.array(sorted(S))


def oracle_set(x, R):
    return np.argsort(-np.abs(x))[:R]


def random_set(N, R, rng):
    return rng.choice(N, R, replace=False)
