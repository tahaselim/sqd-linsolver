import numpy as np, json, warnings; warnings.filterwarnings('ignore')
from common import *
rng=np.random.default_rng(11)
n=10; N=2**n; mu=0.5; A=screened_poisson(n,mu); b=np.zeros(N); src=N//2; b[src]=1.0
x=np.linalg.solve(A.toarray(),b); xh=x/np.linalg.norm(x); PR=1/np.sum(xh**4)
dt=np.pi/4.0; shots=2000; out=dict(N=N,mu=mu,participation_ratio=float(PR))
Rs=[2,4,8,16,32]
for M in [8,32]:
    times=dt*np.arange(1,M+1); P=exact_probs(A,b,times); cnt=sample_counts(P,shots,rng)
    obs=np.nonzero(cnt)[0]; order=np.argsort(-cnt)
    d=dict(t_max=float(times[-1]),R_observed=int(len(obs)),untruncated=metrics(A,x,galerkin(A,b,obs)),
           spread=int(obs.max()-obs.min()+1))
    d['top_R']={R:metrics(A,x,galerkin(A,b,order[:R]))['F'] for R in Rs}
    out[f'M{M}']=d
out['random_R']={R:float(np.mean([metrics(A,x,galerkin(A,b,rng.choice(N,R,replace=False)))['F'] for _ in range(50)])) for R in Rs}
out['random_R_containing_source']={R:float(np.mean([metrics(A,x,galerkin(A,b,np.r_[src,rng.choice(np.setdiff1d(np.arange(N),[src]),R-1,replace=False)]))['F'] for _ in range(50)])) for R in Rs}
out['oracle_R']={R:metrics(A,x,galerkin(A,b,np.argsort(-np.abs(x))[:R]))['F'] for R in Rs}
out['greedy_residual_classical']={R:metrics(A,x,greedy_residual(A,b,R,[src])[0])['F'] for R in Rs}
kry={}
for k in [1,2,4,8,16]:
    S=np.arange(max(0,src-k),min(N,src+k+1)); kry[k]=dict(R=len(S),F=metrics(A,x,galerkin(A,b,S))['F'])
out['classical_krylov_support_k']=kry
# CG cost to e<1e-8
it=[0]; import scipy.sparse.linalg as spla
spla.cg(A,b,rtol=1e-10,callback=lambda xk: it.__setitem__(0,it[0]+1)); out['CG_iterations_1e-10']=it[0]
print(json.dumps(out,indent=1)); json.dump(out,open('t3.json','w'),indent=1)
