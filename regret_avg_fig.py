"""Slide-24 regret figure (fig_regret_avg.png) -- faithful per-round accounting,
shown WITHOUT log-log / "slope" language.

Each epoch: 1 exploration round + L_k=k+1 exploitation rounds at the incumbent.
Sublinear  <=>  average regret per round -> 0  ;  linear <=> it plateaus.

Left panel  (LINEAR axes): cumulative regret -- fixed beta are straight lines
  (linear); cooled beta_k->0 hugs 0, with a zoom inset showing it bends BELOW a
  linear reference (concave = sublinear).
Right panel (x-log, y-lin): average regret per round -- cooled -> 0 (sublinear),
  fixed plateaus at a positive constant (linear).
"""
import sys, os, pickle, random as pyrandom
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "claude_parallelized"))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from meta_algorithm import accepts

def Lk(k): return k + 1

def cum_regret(op, sp, opt, beta_of_k, epochs, p, seeds):
    R = np.zeros(epochs)
    for seed in range(seeds):
        rng = pyrandom.Random(seed); inc = sp[rng.randrange(len(sp))][1]; c = 0.0
        for k in range(epochs):
            b = beta_of_k(k); grp = op if rng.random() < p else sp
            j, cd = grp[rng.randrange(len(grp))]; c += 1 - j[2] / opt
            if accepts(j, inc, b, reduced=True): inc = j
            elif accepts(cd, inc, b, reduced=True): inc = cd
            c += Lk(k) * (1 - inc[2] / opt); R[k] += c
    return R / seeds

if __name__ == "__main__":
    d = pickle.load(open("results_meta/pairs_122.pkl", "rb"))
    opt = d["opt"]; pairs = list(d["pairs"].values())
    op = [pr for pr in pairs if pr[1][2] / opt >= 0.99]
    sp = [pr for pr in pairs if pr[1][2] / opt < 0.99]
    rp = len(op) / len(pairs)
    EP, SEEDS = 12000, 160
    rounds = np.cumsum([1 + Lk(k) for k in range(EP)])
    cur = [(lambda k: 0.1, r"fixed $\beta=0.1$", "tab:red"),
           (lambda k: 0.05, r"fixed $\beta=0.05$", "tab:orange"),
           (lambda k: 0.1/(1+k/60.0), r"cooled $\beta_k\!\to\!0$", "tab:blue")]
    data = [(lbl, col, cum_regret(op, sp, opt, bf, EP, rp, SEEDS)) for bf, lbl, col in cur]
    xr = rounds / 1e6

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.3))
    for lbl, col, R in data: a1.plot(xr, R/1e3, color=col, lw=2.3, label=lbl)
    a1.set_xlabel("cumulative rounds  (millions)"); a1.set_ylabel("cumulative regret  (thousands)")
    a1.set_title("Cumulative regret  (linear axes)"); a1.grid(alpha=0.3); a1.legend(fontsize=9, loc="upper left")
    axin = a1.inset_axes([0.42, 0.40, 0.55, 0.55]); co = data[2][2]
    i0 = np.searchsorted(rounds, 3e5); ref = (co[i0]/rounds[i0]) * rounds
    axin.plot(xr, ref/1e3, "k--", lw=1.2, alpha=0.6, label="linear ref")
    axin.plot(xr, co/1e3, color="tab:blue", lw=2.3, label="cooled")
    axin.set_ylim(0, co[-1]/1e3*1.15); axin.set_title("zoom: cooled bends below linear", fontsize=8.5)
    axin.set_xlabel("rounds (M)", fontsize=8); axin.set_ylabel("regret (k)", fontsize=8)
    axin.tick_params(labelsize=7); axin.grid(alpha=0.3); axin.legend(fontsize=7, loc="upper left")

    for lbl, col, R in data:
        a2.plot(rounds, R/rounds, color=col, lw=2.3, label=lbl + f"  (final {R[-1]/rounds[-1]:.3f})")
    a2.set_xscale("log"); a2.set_ylim(0, 0.09)
    a2.set_xlabel("cumulative rounds"); a2.set_ylabel("average regret per round")
    a2.set_title(r"Average regret per round  (sublinear $\Leftrightarrow\to 0$)")
    a2.grid(alpha=0.3, which="both"); a2.legend(fontsize=9, loc="upper right")
    fig.tight_layout(); fig.savefig("results_meta/fig_regret_avg.png", dpi=150)
    print("saved results_meta/fig_regret_avg.png")
