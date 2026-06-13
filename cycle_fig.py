"""The global optimum is a CYCLE, the trap is a frozen fixed point.

Two small joint-state diagrams for the benchmark game (init (1,2,2)):
  (a) Suboptimal static Nash equilibrium  -- agents freeze at the coordinated profile
      (0,1,2); it self-loops (s'=a=s). g=10 but the private u_i term is starved because
      states never change -> Phi = 13/step ~ 0.87 of optimum.
  (b) Global optimum -- after one transient step the agents enter the period-2 cycle
      (0,1,2) <-> (0,2,1): agents 2 and 3 keep SWAPPING actions (1<->2) while agent 1
      holds 0. Coordination (g=10) is sustained AND states keep moving so u_i is high
      -> Phi = 15/step. The optimum cannot be a fixed point: it requires circulation
      (hence the history buffer H>=1; a memoryless joint policy can only sit at (a)).

Optimal policy from value iteration (find_joint_optimum); potentials from the true Phi.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "claude_parallelized"))
import torch
from parallel_simulation import find_joint_optimum
from congestion_game.reward_functions import make_potential_func

N, S, A, GAMMA, INIT = 3, 3, 3, 0.99, (1, 2, 2)


def phi(state, action):
    Phi = make_potential_func(S)
    return float(Phi(torch.tensor(state), torch.tensor(action)))


def optimal_trajectory(init, steps=6):
    opt = find_joint_optimum(N, S, A, make_potential_func(S), gamma=GAMMA)
    s, traj = tuple(init), [tuple(init)]
    for _ in range(steps):
        s = tuple(int(x) for x in opt[s]); traj.append(s)
    return traj


def node(ax, xy, label, color, r=0.40):
    import matplotlib.patches as mp
    c = mp.Circle(xy, r, facecolor=color, edgecolor="black", lw=1.6, zorder=3)
    ax.add_patch(c)
    ax.text(xy[0], xy[1], label, ha="center", va="center", fontsize=10.5, fontweight="bold", zorder=4)
    return c


def arrow(ax, ca, cb, rad=0.0, color="black", lw=2.0):
    import matplotlib.patches as mp
    ax.add_patch(mp.FancyArrowPatch(ca.center, cb.center, connectionstyle=f"arc3,rad={rad}",
                                    arrowstyle="-|>", mutation_scale=22, lw=lw, color=color,
                                    patchA=ca, patchB=cb, shrinkA=1, shrinkB=1, zorder=2))


def self_loop(ax, c, color, lw=2.0):
    import matplotlib.patches as mp
    x, y = c.center; r = c.radius
    a1 = (x - r * 0.50, y + r * 0.87)            # endpoints ON the top of the circle
    a2 = (x + r * 0.50, y + r * 0.87)
    ax.add_patch(mp.FancyArrowPatch(a2, a1, connectionstyle="arc3,rad=2.2", arrowstyle="-|>",
                                    mutation_scale=16, lw=lw, color=color, zorder=5))


if __name__ == "__main__":
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    traj = optimal_trajectory(INIT)
    trap = (0, 1, 2)
    phi_cyc = phi((0, 1, 2), (0, 2, 1))          # one step inside the optimal cycle
    phi_trap = phi(trap, trap)                   # the frozen fixed point
    print(f"optimal trajectory from {INIT}: {traj}", flush=True)
    print(f"Phi cycle/step = {phi_cyc:.0f} | Phi trap/step = {phi_trap:.0f} | ratio = {phi_trap/phi_cyc:.2f}", flush=True)

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(12, 4.2))

    # ---- (a) trap: frozen fixed point with a self-loop ----
    ca = node(axa, (0, 0), "(0,1,2)", "lightcoral")
    self_loop(axa, ca, "firebrick")
    axa.text(0, 1.05, "stay", ha="center", fontsize=10, color="firebrick", style="italic")
    axa.text(0, -0.95, r"$\bar\Phi=%.0f$/step  $\approx 0.87\,\bar\Phi^\star$" % phi_trap, ha="center", fontsize=11)
    axa.set_title("(a) Suboptimal static NE (trap)\nagents freeze; private reward starved", fontsize=11)
    axa.set_xlim(-1.5, 1.5); axa.set_ylim(-1.3, 1.4); axa.set_aspect("equal"); axa.axis("off")

    # ---- (b) optimum: transient s0 -> period-2 cycle ----
    p0, p1, p2 = (-2.3, 0), (0, 0), (2.3, 0)
    c0 = node(axb, p0, "s$_0$\n(1,2,2)", "lightgray")
    c1 = node(axb, p1, "(0,1,2)", "mediumseagreen")
    c2 = node(axb, p2, "(0,2,1)", "mediumseagreen")
    arrow(axb, c0, c1, rad=0.0, color="gray")                                # transient step
    arrow(axb, c1, c2, rad=0.4, color="darkgreen")                           # cycle: top arc
    arrow(axb, c2, c1, rad=0.4, color="darkgreen")                           # cycle: bottom arc
    axb.text(1.15, 1.18, "agents 2 & 3 swap (1$\\leftrightarrow$2); agent 1 holds 0",
             ha="center", fontsize=9.5, color="darkgreen")
    axb.text(1.15, -1.0, r"$\bar\Phi=%.0f$/step $=\bar\Phi^\star$" % phi_cyc, ha="center", fontsize=11)
    axb.text(-2.3, -0.78, "transient", ha="center", fontsize=9, color="gray", style="italic")
    axb.set_title("(b) Global optimum (period-2 cycle)\ncoordination sustained + states keep moving", fontsize=11)
    axb.set_xlim(-3.2, 3.2); axb.set_ylim(-1.3, 1.4); axb.set_aspect("equal"); axb.axis("off")

    out = "results_meta/fig_cycle.png"; plt.tight_layout(); plt.savefig(out, dpi=150)
    print(f"saved {out}", flush=True)
