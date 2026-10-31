import numpy as np, json, scipy.linalg as sl, warnings; warnings.filterwarnings('ignore')
from common import *
n=6; N=64; A=screened_poisson(n,0.5).toarray(); op=pauli_op(A)
mats=[(c.real,p.to_matrix()) for p,c in zip(op.paulis,op.coeffs)]
b=np.zeros(N); b[N//2]=1; t=2*np.pi; exact=sl.expm(-1j*A*t)@b; res={}
for S in [4,16,64,256]:
    step=np.eye(N,dtype=complex)
    for c,P in mats: step=step@sl.expm(-1j*c*P*t/S)
    psi=np.linalg.matrix_power(step,S)@b
    res[S]=dict(infidelity=float(1-abs(np.vdot(exact,psi))**2),total_cx=332*S); print(S,res[S])
json.dump(res,open('t7.json','w'),indent=1)
