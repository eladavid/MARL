"""Regret view of Nash-hopping. Regret(K)=sum_{k<=K}(1 - Phi(pi^k)/Phi*) = cumulative
foregone welfare of the deployed incumbent vs the global optimum.
  fig_regret_beta.png : per temperature beta (real pool) -- cold beta flattens
                        (sublinear), warm beta stays linear.
  fig_regret_rate.png : per candidate-rate p (cold beta) -- regret ramps ~1/p epochs
                        then flattens => total regret ~ 1/p (bounded cost of rarity).
"""
import sys, os, pickle, random as pyrandom
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "claude_parallelized"))
import numpy as np
from meta_algorithm import accepts


def chain_gap(opt_pairs, sub_pairs, opt, beta, epochs, p, seed):
    """Per-epoch optimality gap (1 - ratio) of the incumbent."""
    rng = pyrandom.Random(seed)
    inc = sub_pairs[rng.randrange(len(sub_pairs))][1]
    gap = np.empty(epochs)
    for k in range(epochs):
        grp = opt_pairs if (rng.random() < p) else sub_pairs
        jump, cand = grp[rng.randrange(len(grp))]
        if accepts(jump, inc, beta, reduced=True):
            inc = jump
        elif accepts(cand, inc, beta, reduced=True):
            inc = cand
        gap[k] = 1.0 - inc[2] / opt
    return gap


def mean_regret(opt_pairs, sub_pairs, opt, beta, epochs, p, seeds):
    g = np.mean([chain_gap(opt_pairs, sub_pairs, opt, beta, epochs, p, s) for s in range(seeds)], axis=0)
    return np.cumsum(g)


if __name__ == "__main__":
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    d = pickle.load(open("results_meta/pairs_122.pkl", "rb"))
    opt = d["opt"]; pairs = list(d["pairs"].values())
    opt_pairs = [pr for pr in pairs if pr[1][2] / opt >= 0.99]
    sub_pairs = [pr for pr in pairs if pr[1][2] / opt < 0.99]
    real_p = len(opt_pairs) / len(pairs)
    print(f"init (1,2,2) | real pool@opt={real_p:.2f}", flush=True)

    # ---- per beta (real pool), log-log with a slope-1 reference -> sublinearity is visible ----
    SEEDS, EP = 300, 1500
    plt.figure(figsize=(7, 5)); x = np.arange(1, EP + 1)
    regs = {}
    for b in [0.3, 0.1, 0.05, 0.02, 0.01]:
        regs[b] = mean_regret(opt_pairs, sub_pairs, opt, b, EP, real_p, SEEDS)
        plt.plot(x, regs[b], label=f"$\\beta$={b}")
    slope = regs[0.3][-1] / EP                          # match the warm (linear) regime
    plt.plot(x, slope * x, "k--", lw=1.0, alpha=0.6, label="linear $\\propto K$ (slope 1)")
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("hopping epoch $K$ (log)"); plt.ylabel("cumulative regret (log)")
    plt.title(r"Sublinear regret at low temperature (init $(1,2,2)$):"
              "\n" r"cold $\beta$ bends below the slope-1 line; warm $\beta$ tracks it")
    plt.legend(fontsize=8, loc="lower right"); plt.grid(alpha=0.3, which="both")
    plt.tight_layout(); plt.savefig("results_meta/fig_regret_beta.png", dpi=150)
    print("saved results_meta/fig_regret_beta.png", flush=True)

    # ---- per rate p (cold beta so nu->1 for all shown p, i.e. regret truly flattens) ----
    BETA = 0.002
    plt.figure(figsize=(7, 5))
    for p in [real_p, 0.05, 0.01, 0.001]:
        ep = max(2000, int(60 / p))
        reg = mean_regret(opt_pairs, sub_pairs, opt, BETA, ep, p, 150)
        lbl = f"p={p:.2f}" if p >= 0.05 else f"p={p:g}"
        plt.plot(np.arange(1, ep + 1), reg, label=lbl + f" (final {reg[-1]:.0f})")
    plt.xscale("log"); plt.xlabel("hopping epoch $K$ (log)")
    plt.ylabel(r"cumulative regret"); plt.grid(alpha=0.3, which="both")
    plt.title(r"Bounded cost of rarity: regret ramps $\sim 1/p$ then flattens ($\beta=0.002$)")
    plt.legend(fontsize=8, loc="upper left")
    plt.tight_layout(); plt.savefig("results_meta/fig_regret_rate.png", dpi=150)
    print("saved results_meta/fig_regret_rate.png", flush=True)
