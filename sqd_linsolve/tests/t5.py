import numpy as np, json, warnings; warnings.filterwarnings('ignore')
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import LieTrotter
from common import *
rng=np.random.default_rng(5); out={}
n=10; N=2**n; mu=0.5; A=screened_poisson(n,mu); b=np.zeros(N); src=N//2; b[src]=1.0
x=np.linalg.solve(A.toarray(),b); dt=np.pi/4; M=8; times=dt*np.arange(1,M+1); P=exact_probs(A,b,times)
# T5 noise: independent readout bit flips
noise={}
for pf in [0.0,1e-3,1e-2,5e-2]:
    r=[]
    for rep in range(5):
        cnt=sample_counts(P,2000,rng,readout_flip=pf,n=n); obs=np.nonzero(cnt)[0]; order=np.argsort(-cnt)
        top=list(order[:16]); topS=list(dict.fromkeys([src]+list(order[:15])))
        r.append((len(obs),metrics(A,x,galerkin(A,b,obs))['Anorm'],metrics(A,x,galerkin(A,b,top))['F'],metrics(A,x,galerkin(A,b,topS))['F']))
    r=np.array(r); noise[pf]=dict(R_untrunc=float(r[:,0].mean()),Anorm_err_untrunc=float(r[:,1].max()),F_top16=float(r[:,2].mean()),F_top16_with_source=float(r[:,3].mean()))
out['T5_readout_noise']=noise; print("T5:",json.dumps(noise,indent=0))
# T6 shot budget (untruncated)
shotsres={}
for s in [20,100,500,2000,20000]:
    rr=[]
    for rep in range(5):
        obs=np.nonzero(sample_counts(P,s,rng))[0]; m=metrics(A,x,galerkin(A,b,obs)); rr.append((len(obs),m['F'],m['rel_err'],src in set(obs)))
    rr=np.array(rr,dtype=float); shotsres[s]=dict(R=float(rr[:,0].mean()),F=float(rr[:,1].mean()),rel_err=float(rr[:,2].mean()),source_captured_frac=float(rr[:,3].mean()))
out['T6_shots']=shotsres; print("T6:",json.dumps(shotsres,indent=0))
# circuit cost of ONE Trotter step for the screened-Poisson class
cost={}
for nn in range(3,8):
    op=pauli_op(screened_poisson(nn,mu).toarray())
    qc=QuantumCircuit(nn); qc.append(PauliEvolutionGate(op,time=0.1,synthesis=LieTrotter(reps=1)),range(nn))
    tq=transpile(qc,basis_gates=['cx','rz','sx','x'],optimization_level=1,seed_transpiler=1)
    cost[nn]=dict(K=len(op),cx_per_step=tq.count_ops().get('cx',0),depth=tq.depth())
out['circuit_cost_one_step']=cost; print("cost:",cost)
json.dump(out,open('t5.json','w'),indent=1)
