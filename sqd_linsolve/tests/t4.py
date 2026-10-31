import numpy as np, json, warnings; warnings.filterwarnings('ignore')
from common import *
rows=[]
for n in range(2,11):
    A=screened_poisson(n,0.5).toarray(); op=pauli_op(A); lam=np.linalg.eigvalsh(A)
    nonI=[abs(c) for l,c in zip(op.paulis,op.coeffs) if str(l)!='I'*n]
    rows.append(dict(n=n,N=2**n,K=len(op),max_weight=max(sum(ch!='I' for ch in str(l)) for l in op.paulis),
                     true_width=float(lam[-1]-lam[0]),bound_2sum=float(2*sum(nonI)))); print(rows[-1])
json.dump(rows,open('t4.json','w'),indent=1)
