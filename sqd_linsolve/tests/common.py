"""Shim: expose the sqd_linsolve package under the names used by the test scripts."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, scipy.linalg as sl, scipy.sparse as sp          # noqa: F401
from sqd_linsolve.core import pauli_op, galerkin, metrics           # noqa: F401
from sqd_linsolve.baselines import exact_probs, sample_counts, greedy_residual, krylov_support  # noqa: F401
from sqd_linsolve.problems import screened_poisson, random_spd, hidden_block_spd, paper_4x4     # noqa: F401
