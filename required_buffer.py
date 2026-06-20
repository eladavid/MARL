"""Trace the MINIMAL buffer needed to REPRESENT the known joint optimum.

We know the optimal joint trajectory (VI). For each agent i, simulate the code's pad-0
history buffer of length H and check whether the map  (H past states, current state) -> action
is single-valued along the optimal length-T trajectory. The minimal conflict-free H is the
buffer agent i actually needs; H* = max_i. This is the TRUE requirement, vs the loose
worst-case prod_{j!=i}|S_j|."""
import itertools, numpy as np

def states_from(traj, s0):
    """state at t: s0 for t=0, previous action otherwise (dynamics s'=a)."""
    return [tuple(s0)] + [tuple(traj[t]) for t in range(len(traj) - 1)]

def min_H_for_agent(state_seq, act_seq, Hmax=16, PAD=-1):
    """smallest H s.t. (last-H states, current state)->action is consistent.
    PAD is the pre-start fill token: -1 = distinct sentinel (fixed), 0 = old buggy pad."""
    Tn = len(act_seq)
    for H in range(0, Hmax + 1):
        seen = {}; ok = True
        buf = [PAD] * H
        for t in range(Tn):
            key = tuple(buf + [state_seq[t]])
            if key in seen and seen[key] != act_seq[t]:
                ok = False; break
            seen[key] = act_seq[t]
            if H > 0:
                buf = buf[1:] + [state_seq[t]]
        if ok:
            return H
    return None  # > Hmax

def required_buffer(traj, s0, n):
    states = states_from(traj, s0)
    Hs = []
    for i in range(n):
        sseq = [states[t][i] for t in range(len(traj))]
        aseq = [traj[t][i] for t in range(len(traj))]
        Hs.append(min_H_for_agent(sseq, aseq))
    return Hs

if __name__ == "__main__":
    print("=== k x k x k coverage-bonus game ===")
    from scale_spike import joint_vi as vi_perm
    for k in [3, 4, 5]:
        star, traj = vi_perm(k, k)
        Hs = required_buffer(traj, [0] * k, k)
        worst = k ** (k - 1)
        hstar = "≥16(pad-0/slot-0 ambig)" if None in Hs else max(Hs)
        print(f"k={k}: required H per agent = {Hs}  -> H* = {hstar}   "
              f"(worst-case bound prod|S_j| = {worst})")

    print("\n=== n>|S| pinned-demand game (n=4,m=3) ===")
    from demand_gap import vi as vi_dem
    for d in [(2, 1, 1), (3, 1, 0), (2, 2, 0)]:
        star, traj = vi_dem(4, 3, np.array(d))
        Hs = required_buffer(traj, [0] * 4, 4)
        hstar = "≥16" if None in Hs else max(Hs)
        print(f"demand {d}: required H per agent = {Hs}  -> H* = {hstar}   "
              f"(worst-case bound prod|S_j| = {3**3})")
