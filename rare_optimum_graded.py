"""Rare-optimum stress test for the GRADED k=5 game (matches results_meta/fig_rare_optimum
style: multi-rate curves, error bars, beta -> 1e-3 saturating at 1.0).

Reuses meta_algorithm.accepts (validated) on the cached k=5 pool stats. Importance-resample
the pool to target optimum-rate p (draw an optimal candidate w.p. p, else suboptimal), run
the selection chain with epochs ~ 1/p so the rare optimum is proposed, measure nu^beta(opt)
with error bars over seeds. Shows selection concentrates on Phi* for ANY p>0 as beta->0."""
import sys, os, pickle, random as pyrandom
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import meta_algorithm as M
from graded_game import closed_form

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim_figs")
K, RHO = 5, 0.7
M.N = K
BETAS = [0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
RATES = [0.06, 0.02, 0.01, 0.001]      # 0.06 ~ the real pool rate; rest importance-sampled
LAB = (30/255, 70/255, 110/255)

def select(opt_c, sub_c, opt_phi, beta, epochs, burn, p, seed):
    rng = pyrandom.Random(seed)
    inc = sub_c[rng.randrange(len(sub_c))]            # start suboptimal (trap)
    at = []
    for k in range(epochs):
        grp = opt_c if rng.random() < p else sub_c
        cand = grp[rng.randrange(len(grp))]
        # validated accept test needs meta_algorithm's pyrandom; seed it per-call deterministically
        if M.accepts(cand, inc, beta, reduced=True):
            inc = cand
        if k >= burn:
            at.append(inc[2] / opt_phi >= 0.99)
    return float(np.mean(at))

if __name__ == "__main__":
    pool = pickle.load(open(os.path.join(HERE, f"pool_stats_k{K}.pkl"), "rb"))
    opt_phi = closed_form(K, RHO)
    opt_c = [c for c in pool if c[2] / opt_phi >= 0.99]
    sub_c = [c for c in pool if c[2] / opt_phi < 0.99]
    print(f"pool {len(pool)} | optimal {len(opt_c)} | suboptimal {len(sub_c)}", flush=True)

    seeds = list(range(20))
    plt.figure(figsize=(7, 5))
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
    print(f"{'rate p':>8} {'epochs':>8} | nu^beta", flush=True)
    for p in RATES:
        epochs = max(3000, int(40 / p)); burn = epochs // 3
        means, stds = [], []
        for b in BETAS:
            nus = [select(opt_c, sub_c, opt_phi, b, epochs, burn, p, s) for s in seeds]
            means.append(np.mean(nus)); stds.append(np.std(nus))
        plt.errorbar(BETAS, means, yerr=stds, marker="o", capsize=3, lw=2,
                     label=f"opt-rate p={p:g}  (epochs={epochs:,})")
        print(f"{p:8g} {epochs:8d} | " + " ".join(f"{m:.2f}" for m in means), flush=True)

    plt.xscale("log"); plt.gca().invert_xaxis(); plt.ylim(0, 1.02)
    plt.xlabel(r"temperature $\beta$  ($\to 0$ to the right)")
    plt.ylabel(r"$\nu^\beta(\Pi^\star)$")
    plt.title(r"Rare-optimum stress test (graded $k=5$): selection copes as $p\to0$ (epochs $\sim 1/p$)")
    plt.grid(alpha=0.3); plt.legend(fontsize=8, loc="lower left")
    out = os.path.join(HERE, "fig_hopping_rare.pdf"); plt.tight_layout(); plt.savefig(out, bbox_inches="tight")
    plt.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"\nsaved {out} (+png)", flush=True)
