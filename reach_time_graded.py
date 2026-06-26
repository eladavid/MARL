"""fig_reach_time-style panel for graded k=5: CDF of epochs-to-first-reach-Phi* vs
candidate rate p (reach-time ~ 1/p). Importance-resamples the cached pool to target p,
uses validated accepts, logs the first epoch the incumbent hits Phi*."""
import sys, os, pickle, random as pyrandom
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import meta_algorithm as M
from graded_game import closed_form

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim_figs")
K, RHO = 5, 0.7; M.N = K
RATES = [0.06, 0.02, 0.01, 0.001]
BETA = 0.005          # cold so reaching (not staying) is the limiter

def first_reach(opt_c, sub_c, opt_phi, p, epochs, seed):
    rng = pyrandom.Random(seed); inc = sub_c[rng.randrange(len(sub_c))]
    for k in range(epochs):
        grp = opt_c if rng.random() < p else sub_c          # draw optimal cand w.p. p
        cand = grp[rng.randrange(len(grp))]
        if M.accepts(cand, inc, BETA, reduced=True): inc = cand
        if inc[2] / opt_phi >= 0.99: return k + 1
    return None

if __name__ == "__main__":
    pool = pickle.load(open(os.path.join(HERE, "pool_stats_k5.pkl"), "rb"))
    opt_phi = closed_form(K, RHO)
    opt_c = [c for c in pool if c[2] / opt_phi >= 0.99]
    sub_c = [c for c in pool if c[2] / opt_phi < 0.99]
    plt.figure(figsize=(7, 4.6)); plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
    for p in RATES:
        epochs = max(2000, int(60 / p))
        reaches = [first_reach(opt_c, sub_c, opt_phi, p, epochs, s) for s in range(200)]
        rr = sorted([r for r in reaches if r is not None]); med = int(np.median(rr)) if rr else -1
        cdf = np.arange(1, len(rr) + 1) / len(reaches)
        plt.step(rr, cdf, where="post", lw=2, label=f"p={p:g}  (median {med})")
    plt.xscale("log"); plt.ylim(0, 1.02)
    plt.xlabel("epochs to first reach $\\Phi^\\star$"); plt.ylabel("cumulative fraction of chains")
    plt.title(r"Time-to-optimum vs candidate rate $p$ (graded $k=5$): reach-time $\sim 1/p$")
    plt.grid(alpha=0.3, which="both"); plt.legend(fontsize=9, loc="lower right")
    out = os.path.join(HERE, "fig_reach_time_k5.pdf"); plt.tight_layout(); plt.savefig(out, bbox_inches="tight")
    print(f"saved {out}")
