"""Rare-optimum stress test (importance-sampled candidate rate).

The meta-algorithm's selection holds for ANY optimal-generation rate p>0 (positive
basin), but the chain must propose the optimum at least once, so the epochs needed
scale ~1/p. Here we reweight the existing candidate pool to a TARGET rate p (draw an
optimal PSGA candidate w.p. p, else a suboptimal one) to mimic p in {0.001,0.01,0.05}
— far below what we could afford to generate directly — and show the chain still
concentrates on the optimum as beta->0, given epochs ~ 1/p.
"""
import sys, os, pickle, random as pyrandom
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "claude_parallelized"))
import numpy as np
from meta_algorithm import accepts

BETAS = [0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]   # cool further for the rare-optimum regime
RATES = [0.18, 0.05, 0.01, 0.001]      # 0.18 ~ the real pool; the rest are importance-sampled


def select_reweighted(opt_pairs, sub_pairs, opt_phi, beta, epochs, burn_in, p, seed):
    """Two-stage selection where an OPTIMAL-cand pair is drawn w.p. p each epoch."""
    rng = pyrandom.Random(seed)
    inc = sub_pairs[rng.randrange(len(sub_pairs))][1]      # start suboptimal
    at_opt = []
    for k in range(epochs):
        grp = opt_pairs if (rng.random() < p) else sub_pairs
        jump, cand = grp[rng.randrange(len(grp))]
        if accepts(jump, inc, beta, reduced=True):
            inc = jump
        elif accepts(cand, inc, beta, reduced=True):
            inc = cand
        if k >= burn_in:
            at_opt.append(inc[2] / opt_phi >= 0.99)
    return float(np.mean(at_opt))


if __name__ == "__main__":
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    d = pickle.load(open("results_meta/pairs_122.pkl", "rb"))
    opt_phi = d["opt"]; pairs = list(d["pairs"].values())
    opt_pairs = [pr for pr in pairs if pr[1][2] / opt_phi >= 0.99]
    sub_pairs = [pr for pr in pairs if pr[1][2] / opt_phi < 0.99]
    print(f"pool: {len(pairs)} pairs | optimal-cand {len(opt_pairs)} | suboptimal {len(sub_pairs)}", flush=True)

    seeds = list(range(20))
    plt.figure(figsize=(7, 5))
    print(f"{'rate p':>8} {'epochs':>8} | nu^beta across beta", flush=True)
    for p in RATES:
        epochs = max(3000, int(40 / p)); burn = epochs // 3      # epochs ~ 1/p so the optimum is reached
        means, stds = [], []
        for b in BETAS:
            nus = [select_reweighted(opt_pairs, sub_pairs, opt_phi, b, epochs, burn, p, s) for s in seeds]
            means.append(np.mean(nus)); stds.append(np.std(nus))
        plt.errorbar(BETAS, means, yerr=stds, marker="o", capsize=3,
                     label=f"opt-rate p={p:g}  (epochs={epochs:,})")
        print(f"{p:8g} {epochs:8d} | " + " ".join(f"{m:.2f}" for m in means), flush=True)

    plt.xscale("log"); plt.gca().invert_xaxis(); plt.ylim(0, 1.02)
    plt.xlabel(r"temperature $\beta$  ($\to 0$ to the right)")
    plt.ylabel(r"$\nu^\beta(\Pi^\star)$")
    plt.title("Rare-optimum stress test: selection copes as $p\\to0$ (epochs $\\sim 1/p$)")
    plt.grid(alpha=0.3); plt.legend(fontsize=8)
    out = "results_meta/fig_rare_optimum.png"; plt.tight_layout(); plt.savefig(out, dpi=150)
    print(f"\nsaved {out}", flush=True)
