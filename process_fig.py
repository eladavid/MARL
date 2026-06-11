"""In-epoch dynamics of the two-stage Nash-hopping algorithm, on a single chain.

Shows, per hopping epoch: the PSGA candidate offered, the incumbent (staircase), and
which mechanism moved it — STAGE-2 early-stop (random hardened jump accepted, no
training) vs STAGE-3/4 PSGA candidate accepted by the second decision. Rejections
leave the incumbent flat. (Faithful two-stage replay over the real candidate pool.)
"""
import sys, os, pickle, random as pyrandom
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "claude_parallelized"))
import numpy as np
from meta_algorithm import accepts

BETA, EPOCHS, SEED = 0.05, 400, 3


def chain_detailed(pairs, opt, beta, epochs, seed):
    rng = pyrandom.Random(seed)
    inc = pairs[rng.randrange(len(pairs))][1]
    inc_r = np.empty(epochs)        # incumbent optimality after each epoch
    cand_r = np.empty(epochs)       # PSGA candidate offered each epoch
    jump_r = np.empty(epochs)       # early-stop (hardened-jump) candidate offered
    event = np.empty(epochs, int)   # 0 reject(stay) | 1 PSGA-accept | 2 early-stop-accept
    for k in range(epochs):
        jump, cand = pairs[rng.randrange(len(pairs))]
        jump_r[k], cand_r[k] = jump[2] / opt, cand[2] / opt
        if accepts(jump, inc, beta, reduced=True):          # Decision 1 (early stop)
            inc, event[k] = jump, 2
        elif accepts(cand, inc, beta, reduced=True):        # Decision 2 (after PSGA)
            inc, event[k] = cand, 1
        else:
            event[k] = 0
        inc_r[k] = inc[2] / opt
    return inc_r, cand_r, jump_r, event


if __name__ == "__main__":
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    d = pickle.load(open("results_meta/pairs_122.pkl", "rb"))
    opt = d["opt"]; pairs = list(d["pairs"].values())
    inc_r, cand_r, jump_r, event = chain_detailed(pairs, opt, BETA, EPOCHS, SEED)
    x = np.arange(EPOCHS)

    plt.figure(figsize=(9, 4.5))
    # candidate streams offered each epoch (what the chain saw)
    plt.scatter(x, cand_r, s=8, c="lightsteelblue", alpha=0.6, label="PSGA candidate offered")
    plt.scatter(x, jump_r, s=6, c="lightgray", alpha=0.5, marker="x", label="hardened-jump offered")
    # incumbent staircase
    plt.step(x, inc_r, where="post", color="black", lw=1.6, label="incumbent")
    # which mechanism moved the incumbent
    es, pa = event == 2, event == 1
    plt.scatter(x[pa], inc_r[pa], s=70, marker="o", facecolors="none", edgecolors="green",
                lw=1.6, label="PSGA accepted (2nd decision)", zorder=5)
    plt.scatter(x[es], inc_r[es], s=90, marker="^", color="orange",
                edgecolors="k", lw=0.5, label="early-stop jump accepted", zorder=6)
    plt.axhline(1.0, ls="--", lw=0.8, color="k", alpha=0.5)
    plt.ylim(0, 1.05); plt.xlabel("hopping epoch")
    plt.ylabel(r"optimality  $\bar\Phi/\bar\Phi^\star$")
    plt.title(f"Two-stage hopping in action  (init $(1,2,2)$, $\\beta={BETA}$): "
              f"{es.sum()} early-stops, {pa.sum()} PSGA-accepts, {(event==0).sum()} rejects")
    plt.legend(fontsize=8, loc="lower right", ncol=2); plt.grid(alpha=0.3)
    out = "results_meta/fig_process.png"; plt.tight_layout(); plt.savefig(out, dpi=150)
    print(f"init (1,2,2) beta={BETA} seed={SEED}: "
          f"early-stops={es.sum()}, PSGA-accepts={pa.sum()}, rejects={(event==0).sum()}", flush=True)
    print(f"saved {out}", flush=True)
