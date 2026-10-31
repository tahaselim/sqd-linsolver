"""Figure 4 of the paper from t3.json, t4.json, t5.json."""
import json, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
t3 = json.load(open("t3.json")); t5 = json.load(open("t5.json")); t4 = json.load(open("t4.json"))
fig, ax = plt.subplots(2, 2, figsize=(11, 8.2)); Rs = [2, 4, 8, 16, 32]; g = lambda d: [d[str(R)] for R in Rs]
a = ax[0, 0]
a.plot(Rs, g(t3["oracle_R"]), "k-", lw=3, alpha=.35, label="oracle (largest $|x_z|$)")
a.plot(Rs, g(t3["greedy_residual_classical"]), "s--", c="#2ca02c", label="classical greedy residual")
a.plot(Rs, g(t3["M8"]["top_R"]), "o-", c="#1f77b4", label="SQD, top-$R$ by frequency")
kr = t3["classical_krylov_support_k"]; a.plot([kr[k]["R"] for k in kr], [kr[k]["F"] for k in kr], "^:", c="#9467bd", label="classical Krylov support $A^kb$")
a.plot(Rs, g(t3["random_R_containing_source"]), "d-.", c="#ff7f0e", label="random, source included")
a.plot(Rs, g(t3["random_R"]), "x-", c="#d62728", label="random")
a.set_xscale("log", base=2); a.set_xlabel("subspace size $R$"); a.set_ylabel("fidelity $F$"); a.set_title("(a) Selection strategies, $N=1024$"); a.legend(fontsize=7.5); a.grid(alpha=.3)
a = ax[0, 1]; c = t5["circuit_cost_one_step"]
a.semilogy([int(k) for k in c], [c[k]["cx_per_step"] for k in c], "o-", label="CNOTs per Trotter step")
a.semilogy([r["n"] for r in t4], [r["K"] for r in t4], "s--", label="Pauli terms $L$")
a.set_xlabel("qubits $n$"); a.set_title("(b) Circuit cost, screened Poisson"); a.legend(fontsize=8); a.grid(alpha=.3, which="both")
a = ax[1, 0]; nz = t5["T5_readout_noise"]; xs = [max(float(k), 1e-4) for k in nz]
a.semilogx(xs, [nz[k]["R_untrunc"] for k in nz], "o-", c="#1f77b4"); a.set_ylabel("$R$", color="#1f77b4")
a.set_xlabel("readout bit-flip probability (0 plotted at $10^{-4}$)")
a2 = a.twinx(); a2.semilogy(xs, [nz[k]["Anorm_err_untrunc"] for k in nz], "s--", c="#d62728"); a2.set_ylabel("energy-norm error", color="#d62728")
a.set_title("(c) Readout noise"); a.grid(alpha=.3)
a = ax[1, 1]; sh = t5["T6_shots"]; s = [int(k) for k in sh]
a.loglog(s, [sh[k]["rel_err"] for k in sh], "o-"); a.set_xlabel("shots per time point ($M=8$)"); a.set_ylabel("relative error")
a3 = a.twinx(); a3.semilogx(s, [sh[k]["R"] for k in sh], "s--", c="gray"); a3.set_ylabel("$R$", color="gray")
a.set_title("(d) Shot budget"); a.grid(alpha=.3, which="both")
plt.tight_layout(); plt.savefig("figure1.pdf"); plt.savefig("figure1.png", dpi=160)
