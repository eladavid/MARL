"""Regret of Nash-hopping -- faithful per-round accounting (Algorithm 2).

Each epoch k: 1 EXPLORATION round (play the perturbed candidate) + L_k=k+1
EXPLOITATION rounds (deploy the incumbent). Regret = sum of per-round gaps
(1 - Phi/Phi*), plotted vs cumulative rounds.

Empirical finding (beyond the fixed-beta theory):
  * FIXED beta -> the incumbent keeps a small residual gap (occasional hop off
    the optimum) -> regret is LINEAR (slope ~1), constant shrinks with beta.
  * COOLING beta_k -> 0 -> the incumbent locks onto the optimum -> regret is
    SUBLINEAR (slope < 1).
"""
import sys, os, pickle, random as pyrandom
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "claude_parallelized"))
import numpy as np
from meta_algorithm import accepts

def Lk(k): return k + 1                      # gradual (growing) exploitation window

def chain_regret(op, sp, opt, beta_of_k, epochs, p, seed):
    rng = pyrandom.Random(seed)
    inc = sp[rng.randrange(len(sp))][1]; reg = np.empty(epochs); cum = 0.0
    for k in range(epochs):
        b = beta_of_k(k)
        grp = op if (rng.random() < p) else sp
        jump, cand = grp[rng.randrange(len(grp))]
        cum += 1.0 - jump[2] / opt                       # 1 exploration round
        if accepts(jump, inc, b, reduced=True): inc = jump
        elif accepts(cand, inc, b, reduced=True): inc = cand
        cum += Lk(k) * (1.0 - inc[2] / opt)              # L_k exploitation rounds
        reg[k] = cum
    return reg

def mean_regret(op, sp, opt, beta_of_k, epochs, p, seeds):
    return np.mean([chain_regret(op, sp, opt, beta_of_k, epochs, p, s) for s in range(seeds)], axis=0)


if __name__ == "__main__":
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    d = pickle.load(open("results_meta/pairs_122.pkl", "rb"))
    opt = d["opt"]; pairs = list(d["pairs"].values())
    op = [pr for pr in pairs if pr[1][2] / opt >= 0.99]
    sp = [pr for pr in pairs if pr[1][2] / opt < 0.99]
    rp = len(op) / len(pairs)
    print(f"init (1,2,2) | pool@opt={rp:.2f}", flush=True)

    SEEDS, EP = 300, 3000
    rounds = np.cumsum([1 + Lk(k) for k in range(EP)])
    curves = [
        (lambda k: 0.10, r"fixed $\beta=0.1$",  "tab:red"),
        (lambda k: 0.05, r"fixed $\beta=0.05$", "tab:orange"),
        (lambda k: 0.1/(1+k/60.0), r"cooled $\beta_k\!\to\!0$", "tab:blue"),
    ]
    plt.figure(figsize=(7, 5))
    finals = {}
    for bf, lbl, col in curves:
        R = mean_regret(op, sp, opt, bf, EP, rp, SEEDS); finals[lbl] = R[-1]
        s = np.polyfit(np.log(rounds[EP//2:]), np.log(R[EP//2:]), 1)[0]
        plt.plot(rounds, R, color=col, lw=2, label=lbl + f"  (slope {s:.2f})")
    slope1 = finals[r"fixed $\beta=0.1$"] / rounds[-1]
    plt.plot(rounds, slope1 * rounds, "k--", lw=1, alpha=0.6, label="linear (slope 1)")
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel(r"cumulative rounds  $\sum_k(1+L_k)$"); plt.ylabel("cumulative regret (log)")
    plt.title(r"Fixed $\beta$: linear regret (residual gap)."
              "\n" r"Cooling $\beta_k\!\to\!0$: sublinear (incumbent locks at the optimum).")
    plt.legend(fontsize=9, loc="lower right"); plt.grid(alpha=0.3, which="both")
    plt.tight_layout(); plt.savefig("results_meta/fig_regret_beta.png", dpi=150)
    print("saved fig_regret_beta.png |", {k: round(v) for k, v in finals.items()}, flush=True)
