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


"""Diagnostics used in the paper: circuit cost, Trotter accuracy, exact-propagator sampling,
and the statistics of Sec. "Validation on simulator and hardware" (pooled standard error, TVD)."""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from qiskit import transpile
from qiskit.quantum_info import Statevector

from .core import build_evolution_circuit
from .baselines import exact_probs, sample_counts

__all__ = ["BASIS_GATES", "circuit_cost", "trotter_infidelity", "choose_trotter_steps",
           "sample_exact", "pooled_sigma", "tvd", "shot_noise_tvd", "counts_per_time"]

BASIS_GATES = ["cx", "rz", "sx", "x"]


def circuit_cost(op, b, t, trotter_steps, optimization_level=1, seed_transpiler=1):
    """(CNOT count, depth) of the measured circuit after transpilation to {CX, RZ, SX, X}."""
    qc = transpile(build_evolution_circuit(op, b, t, trotter_steps), basis_gates=BASIS_GATES,
                   optimization_level=optimization_level, seed_transpiler=seed_transpiler)
    return qc.count_ops().get("cx", 0), qc.depth()


def trotter_infidelity(op, A, b, t, trotter_steps):
    """1 - |<psi_exact(t)|psi_Trotter(t)>|^2. The circuit is transpiled (optimization level 0)
    so that the Lie-Trotter synthesis is applied gate by gate."""
    qc = transpile(build_evolution_circuit(op, b, t, trotter_steps, measure=False),
                   basis_gates=BASIS_GATES, optimization_level=0)
    sv = Statevector(qc).data
    psi0 = np.asarray(b, complex) / np.linalg.norm(b)
    ex = spla.expm_multiply(-1j * sp.csc_matrix(A, dtype=complex) * t, psi0)
    return float(1 - abs(np.vdot(ex, sv)) ** 2)


def choose_trotter_steps(op, A, b, t_max, tol=1e-2, s_max=64):
    """Smallest S = 1, 2, 4, ... with Trotter infidelity below tol at t_max (capped at s_max).
    Returns (S, infidelity)."""
    S = 1
    while True:
        inf = trotter_infidelity(op, A, b, t_max, S)
        if inf < tol or S >= s_max:
            return S, inf
        S *= 2


def sample_exact(A, b, times, shots, seed=0, readout_flip=0.0):
    """Aggregated counts from the exact distributions p(z|t_j) (no circuits, no Trotter error)."""
    n = int(np.log2(len(b)))
    return sample_counts(exact_probs(A, b, times), shots, np.random.default_rng(seed),
                         readout_flip=readout_flip, n=n)


def pooled_sigma(P, shots):
    """Standard error of the pooled estimate p_agg_hat for M independent circuits with
    exact distributions P (shape (M, N)): sigma_z = sqrt(sum_j p_j (1 - p_j)) / (M sqrt(shots))."""
    P = np.asarray(P)
    return np.sqrt((P * (1 - P)).sum(0)) / (P.shape[0] * np.sqrt(shots))


def tvd(p, q):
    """Total variation distance 1/2 sum_z |p(z) - q(z)|."""
    return 0.5 * float(np.abs(np.asarray(p) - np.asarray(q)).sum())


def shot_noise_tvd(sigma):
    """Expected TVD of an ideal sampler: 1/2 sqrt(2/pi) sum_z sigma_z."""
    return 0.5 * np.sqrt(2 / np.pi) * float(np.sum(sigma))


def counts_per_time(counts_list, N):
    """List of Qiskit counts dictionaries -> (M, N) integer array."""
    out = np.zeros((len(counts_list), N), dtype=np.int64)
    for j, c in enumerate(counts_list):
        for key, v in c.items():
            out[j, int(key.replace(" ", ""), 2)] += v
    return out
