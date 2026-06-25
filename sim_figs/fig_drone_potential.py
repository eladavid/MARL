"""Drone game: blind coordination ascends the shared potential.
Phi/Phi* vs MAC-REINFORCE episode for several seeds (H=H*=2). Shows independent
per-agent learning monotonically climbing the shared potential (the theorem), with
bimodal outcomes; the static-parking trap and the global optimum are marked."""
import os, pickle
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
LAB = (30/255, 70/255, 110/255); GOLD = (176/255, 124/255, 38/255); RED = (0.7, 0.2, 0.15)
d = pickle.load(open(os.path.join(HERE, "drone_traj.pkl"), "rb"))
curves = d["curves"]; EP = d["EP"]; trap = d["trap"]
x = np.arange(len(next(iter(curves.values())))) * 100
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(7, 4.3))
finals = {s: c[-1] for s, c in curves.items()}
for s, c in curves.items():
    col = LAB if finals[s] >= 0.9 else RED
    ax.plot(x, c, color=col, lw=1.8, alpha=0.85)
ax.axhline(1.0, ls="--", color=GOLD, lw=1.3, label=r"global optimum $\Phi^\star$")
ax.axhline(trap, ls=":", color="0.5", lw=1.3, label=f"static-parking trap ({trap:.2f})")
ax.plot([], [], color=LAB, lw=1.8, label=r"reaches near-optimum")
ax.plot([], [], color=RED, lw=1.8, label="stalls below")
ax.set_xlabel("MAC-REINFORCE episode"); ax.set_ylabel(r"hardened potential $\Phi/\Phi^\star$")
ax.set_ylim(min(0, min(min(c) for c in curves.values()) - 0.05), 1.05)
nreach = sum(f >= 0.9 for f in finals.values())
ax.set_title(f"Drone game: blind coordination ascends the shared potential (n=3, H={1})\n"
             f"independent per-agent learning, no communication", fontsize=10)
ax.legend(fontsize=8.5, loc="lower right")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_drone_potential.pdf"), bbox_inches="tight", pad_inches=0.03)
print(f"saved fig_drone_potential.pdf | finals={ {s: round(f,2) for s,f in finals.items()} }")
