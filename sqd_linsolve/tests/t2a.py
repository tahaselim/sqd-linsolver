import numpy as np, json, warnings, time; warnings.filterwarnings('ignore')
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import LieTrotter
from qiskit_aer import AerSimulator
from common import *
rng=np.random.default_rng(7)
n=4; N=16; h=N//2
Q1=rng.random((h,h)); Q2=rng.random((h,h))
B=np.zeros((N,N)); B[:h,:h]=Q1@Q1.T+h*np.eye(h); B[h:,h:]=Q2@Q2.T+h*np.eye(h)
perm=rng.permutation(N); A=B[np.ix_(perm,perm)]
src=int(np.where(perm<h)[0][0]); b=np.zeros(N); b[src]=1.0; x=np.linalg.solve(A,b)
SK=set(); v=b.copy()
for k in range(N): SK|=set(np.nonzero(np.abs(v)>1e-12)[0]); v=A@v
lam=np.linalg.eigvalsh(A); dt=np.pi/(lam[-1]-lam[0]); M=8; times=dt*np.arange(1,M+1)
cnt=sample_counts(exact_probs(A,b,times),2000,rng); C=np.nonzero(cnt)[0]
res=dict(N=N,krylov_support=len(SK),exact_dynamics=dict(R=int(len(C)),inside_SK=bool(set(C)<=SK),**metrics(A,x,galerkin(A,b,C))))
op=pauli_op(A); res['K']=len(op); sim=AerSimulator(seed_simulator=3)
for S in [2,8]:
    t0=time.time(); qcs=[]
    for t in times:
        qc=QuantumCircuit(n); qc.x([q for q in range(n) if (src>>q)&1])
        qc.append(PauliEvolutionGate(op,time=t,synthesis=LieTrotter(reps=S)),range(n)); qc.measure_all(); qcs.append(qc)
    r=sim.run(transpile(qcs,sim),shots=2000).result(); agg=set()
    for i in range(M): agg|={int(k,2) for k in r.get_counts(i)}
    Ct=sorted(agg); res[f'trotter_S{S}']=dict(R=len(Ct),spurious_outside_SK=len(set(Ct)-SK),sec=round(time.time()-t0,1),**metrics(A,x,galerkin(A,b,Ct)))
print(json.dumps(res,indent=0)); json.dump(res,open('t2a.json','w'),indent=1)
