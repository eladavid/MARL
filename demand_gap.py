"""Does a PINNED non-uniform demand restore the representation gap in the n>|S| game?
(user's idea: some zones need >1 agent.)

g rewards matching a target distribution d=(d_0,..,d_{m-1}), Sum d = n, PINNED to slots
(so slot 0 must hold d_0 agents -- breaks the slot-permutation symmetry that let pure
congestion rotate freely). u = move-reward. From the collided depot, holding a non-uniform
target while moving forces agents to TAKE TURNS staying on the heavy slot -> an agent must
leave the same slot differently at different times -> representation gap.

For n=4,m=3 we compute Phi*(VI), best memoryless (H=0) by exhaustive enumeration, the gap,
and the optimal trajectory (to see the take-turns cycle). Compared against the symmetric
pure-congestion baseline (gap ~0)."""
import itertools, time, numpy as np
from scale_cong import GAMMA, T

W, C = 10.0, 3.0

def phi_d(state, action, n, m, d):
    a = np.asarray(action); counts = np.bincount(a, minlength=m)
    g = W * (1.0 - np.sum(((counts - d) / n) ** 2))     # max (=W) when counts == d
    u = C * np.sum(np.asarray(state) != a)
    return g + u

def vi(n, m, d):
    states = list(itertools.product(range(m), repeat=n)); idx = {s: k for k, s in enumerate(states)}
    Phi = np.array([[phi_d(s, a, n, m, d) for a in states] for s in states])
    V = np.zeros(len(states)); argA = np.zeros((T, len(states)), dtype=int)
    for t in range(T - 1, -1, -1):
        Q = Phi + GAMMA * V[np.newaxis, :]; argA[t] = Q.argmax(1); V = Q.max(1)
    s0 = tuple([0] * n); s = s0; traj = []
    for t in range(T):
        a = states[argA[t, idx[s]]]; traj.append(a); s = a
    return V[idx[s0]], traj

def best_H0(n, m, d, s0):
    per = list(itertools.product(range(m), repeat=m)); best = -1e9
    for prof in itertools.product(per, repeat=n):
        s = s0; V = 0.0
        for t in range(T):
            a = tuple(prof[i][s[i]] for i in range(n)); V += (GAMMA ** t) * phi_d(s, a, n, m, d); s = a
        if V > best: best = V
    return best

if __name__ == "__main__":
    n, m = 4, 3
    for d in [(2, 1, 1), (3, 1, 0), (2, 2, 0)]:
        d = np.array(d)
        t0 = time.time()
        star, traj = vi(n, m, d); h0 = best_H0(n, m, d, tuple([0] * n)); gap = 1 - h0 / star
        # per-step count vector along the optimal tail (does it hold the heavy slot?)
        tail = [tuple(np.bincount(np.array(a), minlength=m)) for a in traj[-4:]]
        print(f"demand d={tuple(d)}: Phi*={star:.2f} bestH0={h0:.2f}  GAP={gap:.3f}  "
              f"({time.time()-t0:.0f}s)")
        print(f"   opt tail actions: {traj[-4:]}")
        print(f"   opt tail loads:   {tail}   (should hold {tuple(d)})")
