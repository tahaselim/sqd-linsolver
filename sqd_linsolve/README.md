# sqd_linsolve

Companion code to **T. Selim, "Extending Sample-based Quantum Diagonalization to Systems of Linear Equations."**
Sample-based (Krylov) quantum diagonalization for linear systems `A x = b`: the time evolution
`exp(-iA t_j)|b>` is sampled on a quantum computer (or simulator), and the system is solved classically
by a Galerkin projection onto the sampled computational-basis configurations.

## Installation
```bash
pip install -r requirements.txt        # or: pip install -e ".[notebook]"
```
Tested with Qiskit 2.3.0 and 2.5.2, Qiskit Aer 0.17.2, Python 3.12 (identical results).

## Quick start
```python
import numpy as np
import sqd_linsolve as sq

A, b = sq.paper_4x4()
x, info = sq.sqd_solve(A, b, sq.BackendRunner("simulator", seed=42), M=8, dt=0.25, shots=2000, trotter_steps=4)
print(info["R"], sq.metrics(A, np.linalg.solve(A, b), x))
```
For IBM Quantum hardware, create an authenticated `QiskitRuntimeService` with the token read from an
environment variable (never hard-code it), e.g.
```python
import os
from qiskit_ibm_runtime import QiskitRuntimeService
service = QiskitRuntimeService(channel="ibm_quantum_platform", token=os.environ["IBM_CLOUD_TOKEN"])
runner = sq.BackendRunner("hardware", service=service, backend_name="ibm_fez", dynamical_decoupling=True, twirling=True)
```

## Package layout
Step numbers refer to the workflow of Fig. 1 of the paper.

| Module | Content |
|---|---|
| `sqd_linsolve/core.py` | `pauli_op` (Step 1), `spectral_width_bound` (Step 2, Gershgorin or Pauli-sum bound), `build_evolution_circuit` (Step 3), `sample_configurations` (Algorithm 1, Step 4), `configuration_set` (Step 5, always includes supp(b)), `galerkin` (Step 6, pseudo-inverse/Tikhonov for indefinite A), `metrics` (Step 7), `sqd_solve` (end to end) |
| `sqd_linsolve/backends.py` | `BackendRunner`: noiseless simulator, noisy simulator, IBM hardware; all M circuits in one batched job |
| `sqd_linsolve/baselines.py` | exact-propagator distributions and sampling (with optional readout bit flips), classical Krylov support, classical greedy residual selection, oracle and random sets |
| `sqd_linsolve/diagnostics.py` | circuit cost, Trotter infidelity, adaptive choice of the Trotter steps, exact sampling with a seed, pooled standard error, total variation distance |
| `sqd_linsolve/problems.py` | 4x4 test matrix, screened Poisson operator (with a point source), dense random SPD matrices (dense b or point source), hidden block matrix |

## Reproducing the paper
**`reproduce_paper.ipynb`** reproduces every simulator result of the paper with this package and compares each
number with the value printed in the paper (144 checks; about 10 minutes, most of it for the dense ensemble of Table VI).
The scripts below are the original command-line versions; run them from `tests/` (the example from `examples/`).

| Paper | Notebook section | Script |
|---|---|---|
| Sec. IV A, Fig. 2, Tables I–II (4x4 example, Aer seed 42) | 2 | `examples/example_4x4.py`, `tests/t1.py` |
| Table III, Fig. 3 (simulators; hardware from stored counts) | 2 | – |
| Sec. IV B, Tables IV–V (screened Poisson, N = 1024) | 3 | `tests/t3.py` (Fig. 4(a); seed 11) |
| Sec. IV C: readout noise, shots (Fig. 4(c)–(d)) | 4 | `tests/t5.py` |
| Sec. IV C: Proposition 1 with circuits (hidden blocks) | 4 | `tests/t2a.py` |
| Remarks 1 and 2 | 1 | `tests/t2b.py` |
| Sec. IV D: Pauli terms, CNOTs per Trotter step, Trotter steps at n = 6 (Fig. 4(b)) | 5 | `tests/t4.py`, `tests/t5.py`, `tests/t7.py` |
| Table VI: dense random SPD matrices | 5 | `tests/t8.py` |
| Sec. IV E, Fig. 5: scaling with N | 6 | – |
| Sec. IV F: classical Krylov support, conjugate gradients | 7 | `tests/t3.py` |
| Fig. 4 | 7 | `tests/make_figure.py` |

Tables IV–V, the seed statistics of Sec. IV B and Fig. 5 use the exact-sampling seed 0 of the self-contained
notebook; `tests/t3.py` uses its own seed (11), which gives the same subspaces and fidelities but R = 111 instead of
109 for M = 32.

**Hardware.** The `ibm_fez` results (Fig. 3, Table III) need an IBM Quantum account to run. The self-contained
notebook exports the per-time counts of the paper's job to `data/ibm_fez_4x4_counts.json`; with that file present,
`reproduce_paper.ipynb` also reproduces the hardware row of Table III.

## License
MIT License, see `LICENSE`. Copyright (c) 2026 Taha Selim.
