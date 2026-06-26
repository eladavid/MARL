"""fig_psga-style panel for k=5: individual MAC-REINFORCE runs climb trap-to-trap;
most stall, few reach Phi*. Reads pool_traj_k5.pkl (per-candidate Phi/opt every 100 ep)."""
import os, pickle
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
LAB = (30/255, 70/255, 110/255); GOLD = (176/255, 124/255, 38/255); RED = (0.7, 0.2, 0.15); GRAY = "0.8"
d = pickle.load(open(os.path.join(HERE, "pool_traj_k5.pkl"), "rb"))
traj = np.array(d["traj"]); ratios = np.array(d["ratios"])      # (Q, 50), (Q,)
EP_step = 100; x = np.arange(1, traj.shape[1] + 1) * EP_step
optfrac = float((ratios >= 0.99).mean())

plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(7, 4.3))
# faint: all runs
for q in range(traj.shape[0]):
    ax.plot(x, traj[q], color=GRAY, lw=0.4, alpha=0.3, zorder=1)
# highlight a few reachers (green/blue) and stallers (red)
reach = np.where(ratios >= 0.99)[0]; stall = np.where(ratios < 0.9)[0]
rng = np.random.RandomState(0)
for j, q in enumerate(rng.choice(reach, size=min(3, len(reach)), replace=False)):
    ax.plot(x, traj[q], color=LAB, lw=2, zorder=3, label="reaches $\\Phi^\\star$" if j == 0 else None)
for j, q in enumerate(rng.choice(stall, size=min(3, len(stall)), replace=False)):
    ax.plot(x, traj[q], color=RED, lw=2, zorder=3, label="stalls at a trap" if j == 0 else None)
ax.axhline(1.0, ls="--", color=GOLD, lw=1.2, label="global optimum $\\Phi^\\star$")
ax.set_xlabel("MAC-REINFORCE episode"); ax.set_ylabel(r"learned-policy potential $\Phi/\Phi^\star$")
ax.set_ylim(0.4, 1.05)
ax.set_title(f"Individual MAC-REINFORCE runs climb trap-to-trap (k=5)\n"
             f"only {optfrac:.0%} of {traj.shape[0]} runs reach $\\Phi^\\star$ — selection is needed", fontsize=10)
# dedup legend
h, l = ax.get_legend_handles_labels(); seen = dict(zip(l, h))
ax.legend(seen.values(), seen.keys(), fontsize=9, loc="lower right")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_psga_k5.pdf"), bbox_inches="tight", pad_inches=0.03)
print(f"saved fig_psga_k5.pdf | optfrac={optfrac:.3f} mean={ratios.mean():.3f}")
