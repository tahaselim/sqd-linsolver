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


"""Core SQD pipeline for A x = b (Algorithm 1 and Steps 1-7 of Fig. 1 of the paper)."""
import numpy as np
import scipy.sparse as sp
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import LieTrotter

__all__ = ["pauli_op", "spectral_width_bound", "build_evolution_circuit",
           "sample_configurations", "configuration_set", "galerkin", "metrics", "sqd_solve"]


# ---------------- Step 1: Pauli decomposition ----------------
def pauli_op(A, atol=1e-12):
    """Exact Pauli decomposition A = sum_k c_k P_k (tensorized algorithm, O(N^2 log N)).
    Labels follow Qiskit's little-endian convention (rightmost character = qubit 0)."""
    A = A.toarray() if sp.issparse(A) else np.asarray(A)
    return SparsePauliOp.from_operator(A.astype(complex)).simplify(atol=atol)


# ---------------- Step 2: time grid ----------------
def spectral_width_bound(A, method="gershgorin", op=None):
    """Upper bound on lambda_max - lambda_min.
    'gershgorin': tight and cheap for sparse A (recommended).
    'pauli'     : 2 * sum_{P != I} |c_P| (looser; grows with the number of terms)."""
    if method == "pauli":
        op = pauli_op(A) if op is None else op
        n = op.num_qubits
        return 2.0 * sum(abs(c) for p, c in zip(op.paulis, op.coeffs) if str(p) != "I" * n)
    A = sp.csr_matrix(A)
    d = A.diagonal().real
    r = np.asarray(abs(A).sum(axis=1)).ravel() - np.abs(d)
    return float((d + r).max() - (d - r).min())


# ---------------- Step 3: circuits (state preparation + Trotterized evolution) ----------------
def build_evolution_circuit(op, b, t, trotter_steps, measure=True):
    """Prepare |b>/||b||, apply the S-step Lie-Trotter approximation of exp(-iAt), measure.
    Computational-basis sources are prepared with X gates (no generic state preparation)."""
    n = op.num_qubits
    b = np.asarray(b, dtype=complex)
    qc = QuantumCircuit(n)
    nz = np.flatnonzero(np.abs(b) > 1e-14)
    if len(nz) == 1:
        bits = [q for q in range(n) if (int(nz[0]) >> q) & 1]
        if bits:                      # b = |0...0> needs no X gate
            qc.x(bits)
    else:
        qc.initialize(b / np.linalg.norm(b), range(n))
    qc.append(PauliEvolutionGate(op, time=t, synthesis=LieTrotter(reps=trotter_steps)), range(n))
    if measure:
        qc.measure_all()
    return qc


# ---------------- Steps 4-5: run & sample, configuration set ----------------
def sample_configurations(op, b, times, shots, trotter_steps, runner):
    """Algorithm 1: build all M circuits, submit ONE batched job, return aggregated counts
    as an integer array indexed by computational basis state (length N)."""
    circuits = [build_evolution_circuit(op, b, t, trotter_steps) for t in times]
    counts = runner.run_batch(circuits, shots)
    agg = np.zeros(2 ** op.num_qubits, dtype=np.int64)
    for c in counts:
        for key, v in c.items():
            agg[int(key.replace(" ", ""), 2)] += v
    return agg


def configuration_set(counts, b, max_R=None):
    """Configuration set C (Eq. (config-set) of the paper) = observed configurations, always including supp(b).
    If max_R is given, keep supp(b) plus the most frequently observed configurations."""
    src = [int(i) for i in np.flatnonzero(np.abs(b) > 1e-14)]
    order = [int(i) for i in np.argsort(-counts, kind="stable") if counts[i] > 0]
    C = list(dict.fromkeys(src + order))
    if max_R is not None:
        C = C[:max(max_R, len(src))]
    return np.array(sorted(C))


# ---------------- Step 6: Galerkin solve ----------------
def galerkin(A, b, C, rcond=1e-10, tikhonov=None):
    """x_approx = P A_sub^{-1} P^T b (Step 6). SPD submatrices are solved directly;
    otherwise a spectral pseudo-inverse (or Tikhonov filter) is used (Step 6)."""
    C = np.array(sorted(set(int(i) for i in C)))
    A = sp.csr_matrix(A)
    As = A[C][:, C].toarray()
    bs = np.asarray(b)[C]
    lam, W = np.linalg.eigh(As)
    if lam.min() > rcond * max(1.0, abs(lam).max()):
        xs = np.linalg.solve(As, bs)
    else:
        if tikhonov is not None:
            f = lam / (lam ** 2 + tikhonov ** 2)
        else:
            f = np.where(np.abs(lam) > rcond * abs(lam).max(), 1.0 / np.where(lam == 0, 1, lam), 0.0)
        xs = W @ (f * (W.conj().T @ bs))
    x = np.zeros(A.shape[0], dtype=np.result_type(xs, float))
    x[C] = xs
    return x


# ---------------- Step 7: output metrics ----------------
def metrics(A, x_ref, x):
    """Fidelity and relative error (Eq. (metrics) of the paper), relative residual, energy-norm error."""
    A = sp.csr_matrix(A)
    nr, nx = np.linalg.norm(x_ref), np.linalg.norm(x)
    F = float(abs(np.vdot(x_ref / nr, x / nx)) ** 2) if nx > 0 else 0.0
    e = x_ref - x
    b = A @ x_ref
    return dict(F=F, rel_err=float(np.linalg.norm(e) / nr),
                residual=float(np.linalg.norm(A @ x - b) / np.linalg.norm(b)),
                Anorm=float(np.sqrt(abs(np.vdot(e, A @ e))) / np.sqrt(abs(np.vdot(x_ref, A @ x_ref)))))


# ---------------- end-to-end solver ----------------
def sqd_solve(A, b, runner, M=8, dt=None, shots=2000, trotter_steps=1, max_R=None):
    """Full pipeline. dt defaults to pi / (Gershgorin bound on the spectral width)."""
    op = pauli_op(A)
    if dt is None:
        dt = np.pi / spectral_width_bound(A)
    times = dt * np.arange(1, M + 1)
    counts = sample_configurations(op, b, times, shots, trotter_steps, runner)
    C = configuration_set(counts, b, max_R)
    x = galerkin(A, b, C)
    return x, dict(R=len(C), N=len(b), C=C, counts=counts, L=len(op), K=len(op), dt=dt, times=times)
