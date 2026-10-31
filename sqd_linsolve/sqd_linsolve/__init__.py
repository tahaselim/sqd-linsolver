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


"""sqd_linsolve: sample-based (Krylov) quantum diagonalization for linear systems A x = b.

Companion code to T. Selim, "Extending Sample-based Quantum Diagonalization to
Systems of Linear Equations". Module layout:
  core      -- Pauli decomposition, time step, circuits, sampling, configuration set,
               Galerkin solve, metrics, and the end-to-end solver sqd_solve()
  backends  -- BackendRunner: noiseless simulator, noisy simulator, IBM hardware
  baselines -- classical reference strategies (Krylov support, greedy residual,
               oracle, random) and exact-propagator sampling emulation
  problems  -- test matrices used in the paper
  diagnostics -- circuit cost, Trotter infidelity, exact sampling, pooled standard error, TVD
"""
from .core import *
from .baselines import *
from .problems import *
from .diagnostics import *
from .backends import BackendRunner
__version__ = "1.1.0"
