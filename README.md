

# sqd_linsolve

**Sample-based quantum diagonalization (SQD) for linear systems $Ax = b$**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Companion code to **T. Selim, "Extending Sample-based Quantum Diagonalization to Systems of Linear Equations"** (arXiv, 2026).

The method decomposes a Hermitian matrix $A$ in the Pauli basis and samples the Trotterized time evolution
$e^{-iAt_j}|b\rangle$ at several times on a quantum computer or simulator. It then solves the system classically by a
Galerkin projection onto the sampled computational-basis configurations. The subspace solution is exact when the
sampled configurations contain the Krylov support of $b$. For symmetric positive-definite $A$, it is optimal in the
energy norm.

## Contents

```
.
├── sqd_linsolve/                            the Python package
├── reproduce_paper.ipynb                    reproduces the numerical results of the paper
├── sqd_linear_systems_selfcontained.ipynb   step-by-step implementation, worked example, hardware runs
├── tests/                                   command-line scripts t1–t8 and their stored results
├── examples/example_4x4.py                  minimal end-to-end example
├── data/                                    IBM Quantum counts used in the paper
├── requirements.txt, pyproject.toml
└── LICENSE
```

## Installation

```bash
git clone https://github.com/<user>/sqd_linsolve.git
cd sqd_linsolve
pip install -e ".[notebook]"          # or: pip install -r requirements.txt
```

Requires Python ≥ 3.10, Qiskit ≥ 2.3 and Qiskit Aer ≥ 0.17. Tested with Qiskit 2.3.0 and 2.5.2 and Qiskit Aer 0.17.2,
with identical results. Install `qiskit-ibm-runtime` (`pip install -e ".[hardware]"`) to run on IBM Quantum hardware.

## Quick start

```python
import numpy as np
import sqd_linsolve as sq

A, b = sq.paper_4x4()                                   # 4x4 SPD matrix, b = |11>
runner = sq.BackendRunner("simulator", seed=42)         # or "noisy_simulator", "hardware"
x, info = sq.sqd_solve(A, b, runner, M=8, dt=0.25, shots=2000, trotter_steps=4)
print("R =", info["R"], sq.metrics(A, np.linalg.solve(A, b), x))
```

The pipeline follows the seven steps of the paper:

| Step | Function |
|---|---|
| 1. Pauli decomposition | `pauli_op` |
| 2. Time grid (Gershgorin bound, Nyquist step) | `spectral_width_bound` |
| 3. State preparation and Lie–Trotter evolution | `build_evolution_circuit` |
| 4. Sampling, all circuits in one batched job | `sample_configurations`, `BackendRunner` |
| 5. Configuration set, always including supp(b) | `configuration_set` |
| 6. Galerkin solve (pseudo-inverse or Tikhonov for indefinite A) | `galerkin` |
| 7. Fidelity, relative error, residual, energy-norm error | `metrics` |

`sqd_solve` chains all seven steps. The modules `baselines` (exact sampling, classical Krylov and greedy selection),
`diagnostics` (circuit cost, Trotter infidelity, total variation distance) and `problems` (the test matrices of the paper)
contain the tools used in the numerical tests.

## Notebooks

**`reproduce_paper.ipynb`** reproduces the numerical results of the paper with the package. It recomputes Tables I–VI and
Figs. 2–5 and compares each number with the value printed in the paper (144 checks). The run takes about 10 minutes,
mostly for the dense ensemble of Table VI, and needs no IBM Quantum account.

**`sqd_linear_systems_selfcontained.ipynb`** implements the method from scratch, without importing the package. It walks
through the 4×4 example one step at a time and explains each quantity. It also runs the scaling test on the noiseless
simulator, the noisy simulator and an exact emulation, and includes the IBM Quantum hardware runs and a localized
problem with $N = 1024$. It runs in about 2.5 minutes without hardware.

## IBM Quantum hardware

The hardware cells are switched off by default. To run them, store your API key in an environment variable. Never
write it into a notebook or a file under version control.

```bash
export IBM_CLOUD_TOKEN="..."            # optional: export IBM_CLOUD_INSTANCE="crn:..."
```

```python
import os
from qiskit_ibm_runtime import QiskitRuntimeService
service = QiskitRuntimeService(channel="ibm_quantum_platform", token=os.environ["IBM_CLOUD_TOKEN"])
runner = sq.BackendRunner("hardware", service=service, backend_name="ibm_fez",
                          dynamical_decoupling=True, twirling=True)
```

The per-time counts of the `ibm_fez` jobs used in the paper are stored in `data/`. With these files present, both
notebooks reproduce the hardware results without an account.

## Citation

If you use this code, please cite:

```bibtex
@article{selim2026sqdlinear,
  author  = {Selim, Taha},
  title   = {Extending Sample-based Quantum Diagonalization to Systems of Linear Equations},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```

## License

The code is released under the [MIT License](LICENSE). Copyright (c) 2026 Taha Selim.


