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


"""Test matrices used in the paper."""
import numpy as np
import scipy.sparse as sp

__all__ = ["paper_4x4", "screened_poisson", "screened_poisson_system", "random_spd",
           "random_spd_point", "hidden_block_spd"]


def paper_4x4():
    A = np.array([[5, 3, 1, -2], [3, 5, 2, -1], [1, 2, 5, -3], [-2, -1, -3, 5]], float)
    return A, np.array([0, 0, 0, 1.0])


def screened_poisson(n, mu):
    """A = -Delta_h + mu I (Dirichlet), N = 2^n, as a sparse matrix."""
    N = 2 ** n; off = -np.ones(N - 1)
    return sp.diags([off, (2 + mu) * np.ones(N), off], [-1, 0, 1], format="csr")


def screened_poisson_system(N, mu=0.5):
    """Screened Poisson system of Secs. "Localized problems" and "Scaling with the matrix dimension":
    A = -Delta_h + mu I (N = 2^n) and a point source b = |N/2>."""
    n = int(np.log2(N))
    b = np.zeros(N); b[N // 2] = 1.0
    return screened_poisson(n, mu), b


def random_spd(N):
    """Single dense random SPD instance A = Q Q^T + N I with a dense b, Q_ij, b_i ~ U[0,1), seed N
    (the ensemble of Table "dense" is generated in tests/t8.py)."""
    np.random.seed(N); Q = np.random.rand(N, N)
    return Q @ Q.T + N * np.eye(N), np.random.rand(N)


def random_spd_point(N, seed=None):
    """Dense random SPD matrix A = Q Q^T + N I with a point source b = |0...0> (scaling test,
    Fig. "scalability"); Q_ij ~ U[0,1) from numpy.random.default_rng(N if seed is None else seed)."""
    rng = np.random.default_rng(N if seed is None else seed); Q = rng.random((N, N))
    b = np.zeros(N); b[0] = 1.0
    return Q @ Q.T + N * np.eye(N), b


def hidden_block_spd(n=4, seed=7):
    """Reducible SPD matrix with two interleaved blocks (Proposition 1 test)."""
    rng = np.random.default_rng(seed); N = 2 ** n; h = N // 2
    Q1, Q2 = rng.random((h, h)), rng.random((h, h))
    B = np.zeros((N, N)); B[:h, :h] = Q1 @ Q1.T + h * np.eye(h); B[h:, h:] = Q2 @ Q2.T + h * np.eye(h)
    perm = rng.permutation(N); A = B[np.ix_(perm, perm)]
    src = int(np.flatnonzero(perm < h)[0]); b = np.zeros(N); b[src] = 1.0
    return A, b, perm
