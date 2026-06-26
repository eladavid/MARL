"""Dedicated HOPPING figure: the meta-algorithm in action on the graded k=5 game.
Builds the k=5 pool (validated-accepts replay), logs the INCUMBENT trajectory Phi/Phi*
over hopping epochs for a cold and a warm beta, and plots:
  (left)  incumbent climbs out of the trap and locks near Phi* (cold) vs wanders (warm),
          with rejected/accepted proposals scattered -- the dynamics.
  (right) concentration nu^beta -> 1 as beta cools (vs pool baseline).
Reuses meta_algorithm.accepts (validated). Saves pool stats for cheap re-plotting."""
import sys, os, pickle, time, random as pyrandom
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
torch.set_num_threads(1)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import meta_algorithm as M
from meta_replay import build_pool_stats
from graded_game import closed_form

LAB = (30/255, 70/255, 110/255); GOLD = (176/255, 124/255, 38/255); GRAY = "0.55"
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim_figs")
K, RHO = 5, 0.7
STATS = os.path.join(HERE, f"pool_stats_k{K}.pkl")


def chain_traj(pool, beta, n, opt, epochs=3000, seed=0):
    """Validated-accepts replay; return per-epoch incumbent ratio + proposal ratios/accepts."""
    M.N = n
    st = pyrandom.getstate(); pyrandom.seed(seed)
    inc = min(pool, key=lambda s: s[2])              # start in the deepest trap
    inc_tr, prop_tr, acc_tr = [], [], []
    for e in range(epochs):
        cand = pool[pyrandom.randrange(len(pool))]
        a = M.accepts(cand, inc, beta, reduced=True)
        if a: inc = cand
        inc_tr.append(inc[2] / opt); prop_tr.append(cand[2] / opt); acc_tr.append(a)
    pyrandom.setstate(st)
    return np.array(inc_tr), np.array(prop_tr), np.array(acc_tr)


if __name__ == "__main__":
    opt = closed_form(K, RHO); t0 = time.time()
    if os.path.exists(STATS):
        pool = pickle.load(open(STATS, "rb")); print(f"loaded cached pool ({len(pool)})", flush=True)
    else:
        pool = build_pool_stats(K, RHO, 250, episodes=5000)
        pickle.dump(pool, open(STATS, "wb"))
        print(f"built+cached pool ({len(pool)}, {time.time()-t0:.0f}s)", flush=True)
    ratios = np.array([s[2] / opt for s in pool])
    optfrac = float((ratios >= 0.99).mean())

    # trajectories: one cold (locks at opt), one warm (wanders)
    EP = 3000
    inc_cold, prop_cold, acc_cold = chain_traj(pool, 0.01, K, opt, EP, seed=0)
    inc_warm, _, _ = chain_traj(pool, 0.1, K, opt, EP, seed=0)

    # concentration curve
    betas = [0.3, 0.15, 0.08, 0.04, 0.02, 0.01]
    nu = {b: float(np.mean([np.mean(chain_traj(pool, b, K, opt, 4000, seed=s)[0][400:] >= 0.99)
                            for s in range(8)])) for b in betas}

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.6, 3.1))
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
    # LEFT: dynamics
    rej = ~acc_cold
    axL.scatter(np.arange(EP)[rej], prop_cold[rej], s=6, color=GRAY, alpha=0.25,
                label="rejected proposal")
    axL.plot(inc_warm, color=GOLD, lw=1.3, alpha=0.8, label=r"incumbent (warm $\beta=0.1$)")
    axL.plot(inc_cold, color=LAB, lw=2.0, label=r"incumbent (cold $\beta=0.01$)")
    axL.axhline(1.0, ls="--", color=GOLD, lw=1.0)
    axL.axhline(ratios.min(), ls=":", color=GRAY, lw=1.0)
    axL.set_xlabel("hopping epoch"); axL.set_ylabel(r"incumbent $\Phi/\Phi^\star$")
    axL.set_ylim(0.4, 1.05); axL.set_title("Hopping climbs out of the trap to $\\Phi^\\star$", fontsize=10)
    axL.legend(fontsize=8, loc="lower right")
    # RIGHT: concentration
    bs = list(nu); vs = [nu[b] for b in bs]
    axR.plot(bs, vs, "o-", color=LAB, lw=2, label=r"hopping $\nu^\beta(\Pi^\star)$")
    axR.axhline(optfrac, ls=":", color=GRAY, lw=1.3,
                label=f"no selection (pool frac {optfrac:.2f})")
    axR.axhline(1.0, ls="--", color=GOLD, lw=1.0)
    axR.set_xscale("log"); axR.invert_xaxis()
    axR.set_xlabel(r"temperature $\beta$  (cooling $\rightarrow$)")
    axR.set_ylabel(r"time at $\Phi^\star$"); axR.set_ylim(0, 1.05)
    axR.set_title("Concentration sharpens as $\\beta$ cools", fontsize=10)
    axR.legend(fontsize=8, loc="upper left")
    fig.suptitle(f"Nash-hopping selects the global optimum  (k={K}, only {optfrac:.0%} of "
                 f"independent runs reach it)", fontsize=10.5, y=1.03)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_hopping.pdf"), bbox_inches="tight", pad_inches=0.03)
    print(f"saved sim_figs/fig_hopping.pdf  nu={ {b: round(v,3) for b,v in nu.items()} }")
