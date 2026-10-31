"""Reproduce the 4x4 walkthrough of Sec. "Validation on simulator and hardware": Pauli terms, circuit cost, sampling, solution."""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from qiskit import transpile
from sqd_linsolve import paper_4x4, pauli_op, build_evolution_circuit, sqd_solve, metrics
from sqd_linsolve.backends import BackendRunner

A, b = paper_4x4()
op = pauli_op(A)
print("Pauli terms:", {str(p): float(c.real) for p, c in zip(op.paulis, op.coeffs)})
for S in (1, 4):
    qc = transpile(build_evolution_circuit(op, b, 2.0, S), basis_gates=["cx", "rz", "sx", "x"],
                   optimization_level=1, seed_transpiler=1)
    print(f"S={S}: {qc.count_ops().get('cx', 0)} CNOTs, depth {qc.depth()}")
runner = BackendRunner("simulator", seed=42)
x, info = sqd_solve(A, b, runner, M=8, dt=0.25, shots=2000, trotter_steps=4)   # t_j = 0.25 ... 2.0, S = 4
print("counts:", {format(i, "02b"): int(c) for i, c in enumerate(info["counts"])}, " R =", info["R"])
print("x_approx =", np.round(x, 4), " metrics:", metrics(A, np.linalg.solve(A, b), x))
