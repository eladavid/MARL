"""Two dynamics figures for the meta-algorithm:
  fig_convergence.png : mean fraction-of-chains-at-optimum vs epoch, per beta (real pool)
                        -> the optimum is reached fast and the plateau is nu^beta.
  fig_reach_time.png  : CDF of the first epoch the optimum is hit, across optimal-rates p
                        -> reach-time scales ~1/p (the cost of rarity is mixing time).
Two-stage hopping replayed over the real candidate pool (reweighted to rate p)."""
import sys, os, pickle, random as pyrandom
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "claude_parallelized"))
import numpy as np
from meta_algorithm import accepts


def run_chain(opt_pairs, sub_pairs, opt, beta, epochs, p, seed):
    """Returns (at_opt indicator per epoch, first-reach epoch or `epochs` if never)."""
    rng = pyrandom.Random(seed)
    inc = sub_pairs[rng.randrange(len(sub_pairs))][1]
    at = np.zeros(epochs); first = epochs
    for k in range(epochs):
        grp = opt_pairs if (rng.random() < p) else sub_pairs
        jump, cand = grp[rng.randrange(len(grp))]
        if accepts(jump, inc, beta, reduced=True):
            inc = jump
        elif accepts(cand, inc, beta, reduced=True):
            inc = cand
        if inc[2] / opt >= 0.99:
            at[k] = 1.0
            if first == epochs:
                first = k
    return at, first


if __name__ == "__main__":
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    d = pickle.load(open("results_meta/pairs_122.pkl", "rb"))
    opt = d["opt"]; pairs = list(d["pairs"].values())
    opt_pairs = [pr for pr in pairs if pr[1][2] / opt >= 0.99]
    sub_pairs = [pr for pr in pairs if pr[1][2] / opt < 0.99]
    real_p = len(opt_pairs) / len(pairs)
    print(f"init (1,2,2) | {len(pairs)} pairs | real pool@opt={real_p:.2f}", flush=True)

    # ---- Fig 2: averaged convergence (real pool), per beta ----
    SEEDS_A, EP_A = 300, 250
    plt.figure(figsize=(7, 5)); x = np.arange(EP_A)
    for b in [0.3, 0.1, 0.05, 0.02, 0.01]:
        m = np.mean([run_chain(opt_pairs, sub_pairs, opt, b, EP_A, real_p, s)[0] for s in range(SEEDS_A)], axis=0)
        plt.plot(x, m, label=f"$\\beta$={b}")
    plt.axhline(1.0, ls="--", lw=0.8, color="k", alpha=0.4)
    plt.ylim(0, 1.03); plt.xlabel("hopping epoch")
    plt.ylabel(r"fraction of chains at optimum")
    plt.title(r"Convergence: optimum reached fast; plateau $=\nu^\beta$ (init $(1,2,2)$)")
    plt.legend(fontsize=9, loc="center right"); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("results_meta/fig_convergence.png", dpi=150)
    print("saved results_meta/fig_convergence.png", flush=True)

    # ---- Fig 3: reach-time CDF across optimal-rate p (cold beta so reaching is the limiter) ----
    SEEDS_B, BETA_B = 300, 0.005
    plt.figure(figsize=(7, 5))
    print(f"{'p':>8} {'epochs':>8} {'median':>8} {'90%':>8}", flush=True)
    for p in [real_p, 0.05, 0.01, 0.001]:
        ep = max(2000, int(60 / p))
        firsts = np.array([run_chain(opt_pairs, sub_pairs, opt, BETA_B, ep, p, s)[1] for s in range(SEEDS_B)])
        fs = np.sort(firsts); cdf = np.arange(1, len(fs) + 1) / len(fs)
        lbl = f"p={p:.2f}" if p >= 0.05 else f"p={p:g}"
        plt.plot(fs, cdf, marker=".", ms=3, label=lbl + f" (median {int(np.median(firsts))})")
        print(f"{p:8g} {ep:8d} {int(np.median(firsts)):8d} {int(np.percentile(firsts,90)):8d}", flush=True)
    plt.xscale("log"); plt.ylim(0, 1.02)
    plt.xlabel("epochs to first reach the optimum"); plt.ylabel("cumulative fraction of chains")
    plt.title(r"Time-to-optimum vs candidate rate $p$ (reach-time $\sim 1/p$)")
    plt.legend(fontsize=8, loc="lower right"); plt.grid(alpha=0.3, which="both")
    plt.tight_layout(); plt.savefig("results_meta/fig_reach_time.png", dpi=150)
    print("saved results_meta/fig_reach_time.png", flush=True)
