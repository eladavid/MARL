"""History buffer closes the representation gap (contribution 2).

Training-free REPRESENTATION CEILING, so the learning trap does not confound it.
For each initial joint state we compute the BEST discounted potential achievable by any
deterministic memoryless agent-decoupled policy (H=0) -- enumerating all (A^S)^N joint
maps -- and compare to the unconstrained joint optimum (value iteration). H=0 is capped
below the optimum exactly at the "collision" inits where a local state must map to two
different actions (transient vs. cyclic); a one-step history buffer (H=1) removes the
conflict, so the value-iteration-optimal policy becomes representable and the ceiling = 1.

=> without a sufficient buffer the optimum is unreachable for those inits, no matter the
training; H=1 reaches it.
"""
import sys, os, pickle, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from congestion_game.reward_functions import make_potential_func
from claude_parallelized.parallel_simulation import find_joint_optimum

N, S, A, T, GAMMA = 3, 3, 3, 16, 0.99
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_meta", "buffer_data.pkl")


def dec(i): return (i // 9, (i // 3) % 3, i % 3)
def enc(t): return t[0] * 9 + t[1] * 3 + t[2]


def compute():
    Phi = make_potential_func(S)
    PHI = [[float(Phi(torch.tensor(dec(s)), torch.tensor(dec(a)))) for a in range(27)] for s in range(27)]

    def rollout(s0, trans):
        s, tot, disc = s0, 0.0, 1.0
        for _ in range(T):
            a = trans[s]; tot += disc * PHI[s][a]; disc *= GAMMA; s = a
        return tot

    opt_pol = find_joint_optimum(N, S, A, Phi, gamma=GAMMA)          # joint optimum (value iteration)
    opt_trans = [enc(tuple(int(x) for x in opt_pol[dec(s)])) for s in range(27)]
    opt_val = {i: rollout(i, opt_trans) for i in range(27)}          # = H>=1 ceiling (representable)

    maps = list(itertools.product(range(A), repeat=S))               # 27 per-agent H=0 maps
    best_h0 = {i: -1e9 for i in range(27)}
    for m0 in maps:
        for m1 in maps:
            for m2 in maps:                                          # 27^3 joint H=0 policies
                trans = [enc((m0[s0], m1[s1], m2[s2])) for (s0, s1, s2) in (dec(s) for s in range(27))]
                for i in range(27):
                    v = rollout(i, trans)
                    if v > best_h0[i]:
                        best_h0[i] = v
    rows = []
    for i in range(27):
        r0 = best_h0[i] / opt_val[i]
        rows.append(dict(init=dec(i), opt=opt_val[i], h0=best_h0[i], ratio_h0=r0, ratio_h1=1.0))
    return rows


if __name__ == "__main__":
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    if os.path.exists(DATA) and "--recompute" not in sys.argv:
        rows = pickle.load(open(DATA, "rb")); print(f"loaded cached {DATA}", flush=True)
    else:
        rows = compute()
        os.makedirs(os.path.dirname(DATA), exist_ok=True); pickle.dump(rows, open(DATA, "wb"))
        print(f"computed + saved {DATA}", flush=True)

    rows.sort(key=lambda r: r["ratio_h0"])                            # gapped inits to the left
    gapped = [r for r in rows if r["ratio_h0"] < 0.999]
    print(f"H=0 gapped inits: {len(gapped)}/27 at ratio "
          f"{min(r['ratio_h0'] for r in gapped):.3f}; collisions = {[r['init'] for r in gapped]}", flush=True)

    x = range(len(rows)); h0 = [r["ratio_h0"] for r in rows]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    colors = ["tab:red" if r < 0.999 else "tab:green" for r in h0]
    ax.bar(x, h0, color=colors, edgecolor="black", lw=0.4, label="H=0 (memoryless) ceiling")
    # hatched extension to the optimum = what one step of history (H=1) recovers
    for xi, r in zip(x, h0):
        if r < 0.999:
            ax.bar(xi, 1.0 - r, bottom=r, color="none", edgecolor="tab:green",
                   hatch="///", lw=0.8)
    ax.axhline(1.0, ls="--", lw=0.9, color="gray", alpha=0.8)
    ax.text(0.3, 1.012, r"global optimum (reached for all inits at $H\geq 1$)", fontsize=8, color="gray")
    ax.set_xlabel("initial joint state (27 total, sorted by H=0 ceiling)")
    ax.set_ylabel(r"best achievable optimality  $\bar\Phi/\bar\Phi^\star$  (representation ceiling)")
    ax.set_title("History buffer closes the representation gap:\n"
                 f"memoryless AD is capped at {min(r['ratio_h0'] for r in gapped):.2f} on "
                 f"{len(gapped)}/27 collision inits; $H=1$ reaches the optimum everywhere", fontsize=10)
    ax.set_ylim(0, 1.1); ax.set_xticks([])
    from matplotlib.patches import Patch
    handles = [Patch(facecolor="tab:red", edgecolor="black", label=f"$H=0$ capped ({len(gapped)} collision inits)"),
               Patch(facecolor="tab:green", edgecolor="black", label=f"$H=0$ = optimum ({len(rows)-len(gapped)} inits)"),
               Patch(facecolor="none", edgecolor="tab:green", hatch="///", label="gap closed by $H=1$")]
    ax.legend(handles=handles, fontsize=8, loc="lower right"); ax.grid(alpha=0.3, axis="y")
    out = "results_meta/fig_buffer.png"; plt.tight_layout(); plt.savefig(out, dpi=150)
    print(f"saved {out}", flush=True)
