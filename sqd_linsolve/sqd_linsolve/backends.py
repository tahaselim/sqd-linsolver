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


"""Backend-agnostic execution: noiseless simulator, noisy simulator, IBM Quantum hardware."""
from qiskit import transpile

__all__ = ["BackendRunner"]


class BackendRunner:
    """mode = 'simulator' | 'noisy_simulator' | 'hardware'.
    Hardware access uses an authenticated QiskitRuntimeService; never hard-code tokens,
    e.g. QiskitRuntimeService(channel=..., token=os.environ["IBM_CLOUD_TOKEN"])."""

    def __init__(self, mode="simulator", noise_model=None, backend_name=None, service=None,
                 seed=None, optimization_level=1, dynamical_decoupling=False, twirling=False):
        self.mode, self.optimization_level = mode, optimization_level
        if mode in ("simulator", "noisy_simulator"):
            from qiskit_aer import AerSimulator
            if mode == "noisy_simulator" and noise_model is None:
                raise ValueError("noisy_simulator requires a NoiseModel")
            self.backend = AerSimulator(noise_model=noise_model, seed_simulator=seed)
        elif mode == "hardware":
            from qiskit_ibm_runtime import SamplerV2
            if service is None or backend_name is None:
                raise ValueError("hardware mode requires service and backend_name")
            self.backend = service.backend(backend_name)
            self.sampler = SamplerV2(mode=self.backend)
            self.sampler.options.dynamical_decoupling.enable = dynamical_decoupling
            self.sampler.options.twirling.enable_gates = twirling
            self.sampler.options.twirling.enable_measure = twirling
        else:
            raise ValueError(f"unknown mode {mode!r}")
        self.last_job_id = None

    def transpile(self, circuits):
        return transpile(circuits, self.backend, optimization_level=self.optimization_level)

    def run_batch(self, circuits, shots):
        """Submit all circuits as ONE job; return a list of counts dictionaries."""
        tq = self.transpile(circuits)
        if self.mode == "hardware":
            job = self.sampler.run(tq, shots=shots)
            self.last_job_id = job.job_id()
            res = job.result()
            return [res[i].data.meas.get_counts() for i in range(len(tq))]
        res = self.backend.run(tq, shots=shots).result()
        return [res.get_counts(i) for i in range(len(tq))]
