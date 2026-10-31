import numpy as np, json
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import LieTrotter
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from common import *
A=np.array([[5,3,1,-2],[3,5,2,-1],[1,2,5,-3],[-2,-1,-3,5]],float); b=np.array([0,0,0,1.])
op=pauli_op(A); print("Pauli terms (qiskit little-endian labels):", [(str(l),round(c.real,6)) for l,c in zip(op.paulis,op.coeffs)])
M,tmax,shots=8,2.0,2000; times=np.linspace(tmax/M,tmax,M)
def circ(t,S,measure=True):
    qc=QuantumCircuit(2); qc.initialize(b/np.linalg.norm(b),[0,1])
    qc.append(PauliEvolutionGate(op,time=t,synthesis=LieTrotter(reps=S)),[0,1])
    if measure: qc.measure_all()
    return qc
basis=['cx','rz','sx','x']
res={}
for S in [1,4]:
    tq=[transpile(circ(t,S),basis_gates=basis,optimization_level=1,seed_transpiler=1) for t in times]
    cx=[c.count_ops().get('cx',0) for c in tq]; dp=[c.depth() for c in tq]
    # Trotter accuracy vs exact
    inf=[1-abs(np.vdot(Statevector(circ(t,S,False)).data, sl.expm(-1j*A*t)@b))**2 for t in times]
    res[S]=dict(cx_per_circuit=max(cx),depth=max(dp),max_trotter_infidelity=float(max(inf)))
print("gate counts / Trotter accuracy:", res)
sim=AerSimulator(seed_simulator=42)
tq=transpile([circ(t,4) for t in times],sim)
out=sim.run(tq,shots=shots).result()
agg={}
for i in range(M):
    for k,v in out.get_counts(i).items(): agg[int(k,2)]=agg.get(int(k,2),0)+v
C=sorted(agg); x=np.linalg.solve(A,b); xa=galerkin(A,b,C)
print("aggregated counts:", {format(k,'02b'):agg[k] for k in C}, " R =",len(C))
m=metrics(A,x,xa); m['residual']=float(np.linalg.norm(A@xa-b)); print("metrics:", m)
json.dump(dict(gates=res,counts={format(k,'02b'):agg[k] for k in C},metrics=m),open('t1.json','w'),indent=1)
