import numpy as np, json, warnings; warnings.filterwarnings('ignore')
from common import *
rng=np.random.default_rng(7); out={}
N=32; Q=rng.random((N,N)); A=Q@Q.T+N*np.eye(N); b=rng.random(N); x=np.linalg.solve(A,b); viol=0; trials=2000
for _ in range(trials):
    Csz=rng.integers(2,N-1); C=rng.choice(N,Csz,replace=False)
    extra=rng.choice(np.setdiff1d(np.arange(N),C),rng.integers(1,N-Csz+1),replace=False)
    viol+= metrics(A,x,galerkin(A,b,np.r_[C,extra]))['Anorm']>metrics(A,x,galerkin(A,b,C))['Anorm']+1e-12
out['prop2']=dict(trials=trials,violations=int(viol))
N=8; Q=rng.random((N,N)); A=Q@Q.T+N*np.eye(N); b=rng.random(N); b/=np.linalg.norm(b)
lam,V=np.linalg.eigh(A); beta=V.T@b; pbar=(V**2)@(beta**2)
ts=rng.uniform(0,5000,20000); pavg=np.mean(np.abs((V*beta)@np.exp(-1j*np.outer(lam,ts)))**2,axis=1)
xs=V@(beta/lam); xs=xs**2/np.sum(xs**2)
out['prop3']=dict(max_dev=float(np.max(np.abs(pavg-pbar))),Bhatt_pbar_vs_x=float(np.sum(np.sqrt(pbar*xs))**2))
print(out); json.dump(out,open('t2b.json','w'),indent=1)
