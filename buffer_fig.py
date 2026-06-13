"""A history buffer is necessary to represent the optimum (contribution 2).

Training-free, single illustrative init. We fix one initial joint state that requires a
buffer -- INIT=(1,2,2), a "collision" init where the two cyclic agents start coincident on
a non-anchor state -- and enumerate the FULL set of deterministic memoryless agent-decoupled
policies (H=0): all (A^S)^N = 27^3 of them. For each we compute the discounted episode
potential from INIT and normalize by the unconstrained joint optimum (value iteration).

Every memoryless policy is sub-optimal: the spanned set tops out at ~0.874 Phi*, strictly
below 1. The optimum is the period-2 cycle (0,1,2)<->(0,2,1), which forces some local state
to map to two different actions (transient vs. cyclic) -- impossible without memory. A
one-step buffer (H=1) disambiguates them and makes the optimal policy representable (ratio 1).
"""
import sys, os, pickle, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from congestion_game.reward_functions import make_potential_func
from claude_parallelized.parallel_simulation import find_joint_optimum

N, S, A, T, GAMMA, INIT = 3, 3, 3, 16, 0.99, (1, 2, 2)
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_meta", "buffer_data.pkl")


def dec(i): return (i // 9, (i // 3) % 3, i % 3)
def enc(t): return t[0] * 9 + t[1] * 3 + t[2]


def compute():
    Phi = make_potential_func(S)
    PHI = [[float(Phi(torch.tensor(dec(s)), torch.tensor(dec(a)))) for a in range(27)] for s in range(27)]
    i0 = enc(INIT)

    def rollout(trans):
        s, tot, disc = i0, 0.0, 1.0
        for _ in range(T):
            a = trans[s]; tot += disc * PHI[s][a]; disc *= GAMMA; s = a
        return tot

    opt_pol = find_joint_optimum(N, S, A, Phi, gamma=GAMMA)
    opt_trans = [enc(tuple(int(x) for x in opt_pol[dec(s)])) for s in range(27)]
    opt_val = rollout(opt_trans)                                     # = H>=1 ceiling (representable)

    maps = list(itertools.product(range(A), repeat=S))               # 27 per-agent H=0 maps
    ratios = []
    for m0 in maps:
        for m1 in maps:
            for m2 in maps:                                          # 27^3 joint H=0 policies
                trans = [enc((m0[s0], m1[s1], m2[s2])) for (s0, s1, s2) in (dec(s) for s in range(27))]
                ratios.append(rollout(trans) / opt_val)
    return dict(init=INIT, opt=opt_val, ratios=ratios)


if __name__ == "__main__":
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import numpy as np
    if os.path.exists(DATA) and "--recompute" not in sys.argv:
        d = pickle.load(open(DATA, "rb")); print(f"loaded cached {DATA}", flush=True)
        if d.get("init") != INIT or "ratios" not in d:                # stale (old all-27 format)
            d = compute(); pickle.dump(d, open(DATA, "wb")); print("stale cache -> recomputed", flush=True)
    else:
        d = compute(); os.makedirs(os.path.dirname(DATA), exist_ok=True)
        pickle.dump(d, open(DATA, "wb")); print(f"computed + saved {DATA}", flush=True)

    ratios = np.array(d["ratios"]); best = ratios.max()
    print(f"init {d['init']} | {len(ratios)} memoryless policies | best H=0 ratio {best:.3f} "
          f"| frac optimal {(ratios>=0.999).mean():.3f}", flush=True)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.hist(ratios, bins=60, range=(ratios.min(), 1.0), color="tab:red", alpha=0.8, edgecolor="white", lw=0.3)
    ax.axvspan(best, 1.0, color="gray", alpha=0.15)                  # the unreachable band
    ax.axvline(best, ls="--", lw=1.4, color="firebrick")
    ax.text(best - 0.004, ax.get_ylim()[1] * 0.92, f"best $H{{=}}0$ = {best:.3f}",
            ha="right", fontsize=9, color="firebrick", fontweight="bold")
    ax.axvline(1.0, ls="--", lw=1.4, color="darkgreen")
    ax.text(1.0 - 0.004, ax.get_ylim()[1] * 0.6, "global optimum = 1\n(reached at $H{=}1$)",
            ha="right", fontsize=9, color="darkgreen", fontweight="bold")
    ax.annotate("", xy=(1.0, ax.get_ylim()[1] * 0.4), xytext=(best, ax.get_ylim()[1] * 0.4),
                arrowprops=dict(arrowstyle="<->", color="gray", lw=1.2))
    ax.text((best + 1.0) / 2, ax.get_ylim()[1] * 0.45, "unreachable\nwithout a buffer",
            ha="center", fontsize=8.5, color="dimgray")
    ax.set_xlabel(r"optimality from $s_0=(1,2,2)$:  $\bar\Phi/\bar\Phi^\star$")
    ax.set_ylabel(f"# memoryless ($H{{=}}0$) policies  (of {len(ratios):,})")
    ax.set_title(r"A buffer is necessary: from $s_0=(1,2,2)$ every memoryless policy is sub-optimal"
                 "\n" r"the entire spanned $H{=}0$ set tops out at $0.87\,\bar\Phi^\star$; the optimum needs $H{=}1$",
                 fontsize=10)
    ax.set_xlim(ratios.min(), 1.03); ax.grid(alpha=0.3, axis="y")
    out = "results_meta/fig_buffer.png"; plt.tight_layout(); plt.savefig(out, dpi=150)
    print(f"saved {out}", flush=True)
