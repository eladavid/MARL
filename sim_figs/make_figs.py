"""Generate the toy-experiment simulation figures (vector PDF, lab-blue).
Reads sim_data.pkl (+ hopping_data.pkl). Run from MARL/: python3 sim_figs/make_figs.py"""
import os, pickle
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LAB  = (30/255, 70/255, 110/255)
GOLD = (176/255, 124/255, 38/255)
GRAY = "0.55"
D = pickle.load(open(os.path.join(HERE, "sim_data.pkl"), "rb"))
try:
    Hop = pickle.load(open(os.path.join(HERE, "hopping_data.pkl"), "rb"))
except FileNotFoundError:
    Hop = None
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    fig.savefig(os.path.join(HERE, name), bbox_inches="tight", pad_inches=0.03)
    print("saved", name)


# ---------- FIG 1: cyclic optimum vs static trap (space-time of slot occupancy) ----------
def fig1():
    f = D["fig1"]; traj = np.array(f["traj"]); k = f["k"]; Tn = len(traj)
    fig, (axo, axt) = plt.subplots(1, 2, figsize=(7.2, 2.9))
    cmap = plt.cm.viridis
    # optimum: rows=agents, cols=time, color=slot
    for ax, title, T_arr in [(axo, "Global optimum: rotating patrol", traj),
                             (axt, "Trap: static deployment", np.tile(traj[-1] * 0 + np.arange(k), (Tn, 1)))]:
        ax.imshow(T_arr.T, aspect="auto", cmap=cmap, interpolation="nearest")
        ax.set_xlabel("time step"); ax.set_yticks(range(k))
        ax.set_yticklabels([f"a{i+1}" for i in range(k)]); ax.set_title(title, fontsize=10)
    axo.set_ylabel("agent")
    axo.text(0.5, -0.42, r"full coverage + everyone moves: $\Phi/\mathrm{step}=W+nC$",
             transform=axo.transAxes, ha="center", color=LAB, fontsize=9)
    axt.text(0.5, -0.42, r"covered but frozen: $\Phi/\mathrm{step}=W$ (no movement)",
             transform=axt.transAxes, ha="center", color=GOLD, fontsize=9)
    fig.suptitle("The optimum is a cycle; the trap is a fixed point", fontsize=11, y=1.02)
    fig.tight_layout(); save(fig, "fig1_cycle_vs_trap.pdf")


# ---------- FIG 2: representation gap ----------
def fig2():
    a = D["fig2a"]; b = D["fig2b"]
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.4, 3.0))
    Hs = sorted(a["reach_vs_H"])
    means = [np.mean(a["reach_vs_H"][h]) for h in Hs]
    stds = [np.std(a["reach_vs_H"][h]) for h in Hs]
    axa.bar([str(h) for h in Hs], means, yerr=stds, color=[GRAY] + [LAB]*(len(Hs)-1),
            capsize=4, width=0.6)
    axa.axhline(1.0, ls="--", color=GOLD, lw=1.3, label=r"$\Phi^\star$")
    axa.set_xlabel("history buffer length $H$"); axa.set_ylabel(r"reached $\Phi/\Phi^\star$")
    axa.set_ylim(0, 1.08); axa.set_title(f"Memoryless caps below the optimum (k={a['k']})", fontsize=10)
    axa.legend(loc="lower right", fontsize=9)
    axa.text(0, means[0]+0.04, "memoryless\n(H=0)", ha="center", fontsize=8, color=GRAY)

    ks = sorted(b["reqH"])
    axb.plot(ks, [b["bound"][k] for k in ks], "o-", color=GRAY,
             label=r"worst-case bound $\prod_{j\neq i}|S_j|=k^{k-1}$")
    axb.plot(ks, [b["reqH"][k] for k in ks], "s-", color=LAB, lw=2,
             label=r"actual required $H^\star$ (traced)")
    axb.set_yscale("log"); axb.set_xlabel("number of agents $k$"); axb.set_ylabel("buffer length")
    axb.set_xticks(ks); axb.set_title("Required memory is tiny and flat", fontsize=10)
    axb.legend(fontsize=8.5, loc="center right")
    fig.tight_layout(); save(fig, "fig2_representation_gap.pdf")


# ---------- FIG 3: selection + hopping ----------
def fig3():
    # k=5 is the dramatic, trap-heavy case (mean ~0.82, few reach Phi*); reuse fig4b[5] ratios
    ksel = 5
    r = np.array(D["fig4b"][ksel]["ratios"]); rmean = float(r.mean())
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
    axa = axes[0]
    axa.hist(r, bins=np.linspace(0.4, 1.02, 26), color=LAB, alpha=0.85)
    axa.axvline(rmean, color=GRAY, lw=1.5, ls="-", label=f"mean {rmean:.2f}")
    axa.axvline(1.0, color=GOLD, lw=1.5, ls="--", label=r"$\Phi^\star$")
    axa.set_xlabel(r"reached $\Phi/\Phi^\star$"); axa.set_ylabel("# runs")
    axa.set_title(f"Independent learning traps (k={ksel})", fontsize=10)
    axa.legend(fontsize=9)

    axb = axes[1]
    if Hop is not None:
        betas = Hop["betas"]; nu = [Hop["nu"][b] for b in betas]
        axb.plot(betas, nu, "o-", color=LAB, lw=2, label=r"hopping $\nu^\beta(\Pi^\star)$")
        axb.axhline(Hop["pool_optfrac"], ls=":", color=GRAY, lw=1.3,
                    label=f"pool optimal frac ({Hop['pool_optfrac']:.2f})")
        axb.axhline(1.0, ls="--", color=GOLD, lw=1.0)
        axb.set_xscale("log"); axb.invert_xaxis()
        axb.set_xlabel(r"temperature $\beta$ (cooling $\rightarrow$)")
        axb.set_ylabel(r"time at $\Phi^\star$"); axb.set_ylim(0, 1.08)
        axb.set_xticks([0.1, 0.01]); axb.set_xticklabels([r"$10^{-1}$", r"$10^{-2}$"])
        axb.set_title("Hopping concentrates on the optimum", fontsize=10)
        axb.legend(fontsize=8.5, loc="center left")
    else:
        axb.text(0.5, 0.5, "hopping_data.pkl\nnot found", ha="center", va="center",
                 transform=axb.transAxes, color=GRAY)
    fig.tight_layout(); save(fig, "fig3_selection_hopping.pdf")


# ---------- FIG 4: scaling / tractability ----------
def fig4():
    a = D["fig4a"]; b = D["fig4b"]
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.4, 3.0))
    ks = a["k"]
    axa.plot(ks, a["vi_cost"], "o-", color=GOLD, lw=2, label=r"brute-force VI  $(k^k)^2 T$ (exp.)")
    axa.plot(ks, a["per_agent"], "s-", color=LAB, lw=2, label=r"per-agent table (ours, $\sim k^3$)")
    axa.axvline(a["vi_wall_k"], ls=":", color=GRAY);
    axa.text(a["vi_wall_k"]+0.05, axa.get_ylim()[1], " VI infeasible\n (memory)", fontsize=8,
             color=GRAY, va="top")
    axa.set_yscale("log"); axa.set_xlabel("number of agents $k$"); axa.set_ylabel("cost / table size")
    axa.set_title("Brute force is exponential; per-agent cost polynomial", fontsize=10)
    axa.legend(fontsize=8.5, loc="center right")

    ks2 = sorted(b)
    means = [b[k]["mean"] for k in ks2]; popt = [b[k]["popt"] for k in ks2]
    axb.plot(ks2, means, "o-", color=LAB, lw=2, label="MAC-REINFORCE mean")
    axb.plot(ks2, popt, "^--", color=GOLD, lw=1.8, label=r"frac reaching $\Phi^\star$ ($p$)")
    axb.axhline(1.0, ls="--", color=GRAY, lw=0.8)
    axb.set_xlabel("number of agents $k$"); axb.set_ylabel("ratio / fraction")
    axb.set_xticks(ks2); axb.set_ylim(0, 1.08)
    axb.set_title("Selection hardens with scale", fontsize=10)
    axb.legend(fontsize=8.5, loc="center left")
    fig.tight_layout(); save(fig, "fig4_scaling.pdf")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4()
    print("all figures written to", HERE)
