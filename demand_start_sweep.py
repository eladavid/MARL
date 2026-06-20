"""We get to CHOOSE the fixed start s0. Find the realistic deployment that maximizes
the representation gap in the pinned-demand n>|S| game (n=4, m=3, demand (2,1,1))."""
import itertools, time, numpy as np
from demand_gap import phi_d, W, C
from scale_cong import GAMMA, T

def vi_all(n, m, d):
    states = list(itertools.product(range(m), repeat=n)); idx = {s: k for k, s in enumerate(states)}
    Phi = np.array([[phi_d(s, a, n, m, d) for a in states] for s in states])
    V = np.zeros(len(states))
    for t in range(T - 1, -1, -1):
        V = (Phi + GAMMA * V[np.newaxis, :]).max(1)
    return V, idx

def best_H0(n, m, d, s0):
    per = list(itertools.product(range(m), repeat=m)); best = -1e9
    for prof in itertools.product(per, repeat=n):
        s = s0; V = 0.0
        for t in range(T):
            a = tuple(prof[i][s[i]] for i in range(n)); V += (GAMMA ** t) * phi_d(s, a, n, m, d); s = a
        if V > best: best = V
    return best

if __name__ == "__main__":
    n, m = 4, 3; d = np.array([2, 1, 1])
    V, idx = vi_all(n, m, d)
    cands = [(0,0,0,0),(0,0,0,1),(0,0,1,1),(0,0,1,2),(0,1,1,2),(1,1,2,2),(2,2,2,0),(1,2,2,0),(0,0,2,2),(1,1,1,2)]
    print(f"demand {tuple(d)}, n={n}, m={m}: gap vs chosen start\n{'s0':>14} | {'Phi*':>8} {'bestH0':>8} {'gap':>7}")
    print("-" * 44)
    rows = []
    for s0 in cands:
        t0 = time.time(); h0 = best_H0(n, m, d, s0); star = V[idx[s0]]; gap = 1 - h0 / star
        rows.append((gap, s0)); print(f"{str(s0):>14} | {star:>8.2f} {h0:>8.2f} {gap:>7.3f}   ({time.time()-t0:.0f}s)", flush=True)
    rows.sort(reverse=True)
    print(f"\nBEST start: {rows[1][1] if rows[0][1]==(0,0,0,0) else rows[0][1]}  "
          f"gap={[r for r in rows][0][0]:.3f}  (collided was {[g for g,s in rows if s==(0,0,0,0)][0]:.3f})")
