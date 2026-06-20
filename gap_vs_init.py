"""Does the representation gap depend on the initial state? (user's hypothesis)
For n=4, m=3: for each candidate s0, compute Phi*(s0) via VI and the best memoryless
(H=0) value reachable FROM s0 by exhaustive enumeration. gap = 1 - bestH0/Phi*.
A 'hard' s0 forces an agent to revisit a slot with a different exit -> memoryless fails.
"""
import itertools, time, numpy as np
from scale_cong import phi, GAMMA, T

def full_vi(n, m):
    states = list(itertools.product(range(m), repeat=n))
    idx = {s: k for k, s in enumerate(states)}
    Phi = np.array([[phi(s, a, n, m) for a in states] for s in states])
    V = np.zeros(len(states))
    for t in range(T - 1, -1, -1):
        V = (Phi + GAMMA * V[np.newaxis, :]).max(1)
    return V, idx

def best_H0_from(s0, n, m):
    per_agent = list(itertools.product(range(m), repeat=m))
    best = -1e9
    for profile in itertools.product(per_agent, repeat=n):
        s = s0; V = 0.0
        for t in range(T):
            a = tuple(profile[i][s[i]] for i in range(n))
            V += (GAMMA ** t) * phi(s, a, n, m); s = a
        if V > best: best = V
    return best

if __name__ == "__main__":
    n, m = 4, 3
    V, idx = full_vi(n, m)
    candidates = [(0,0,0,0), (0,0,0,1), (0,0,1,1), (0,1,1,2), (1,2,2,2), (0,0,1,2), (1,1,2,2)]
    print(f"n={n}, m={m}: representation gap vs initial state\n")
    print(f"{'s0':>14} | {'Phi*':>8} {'bestH0':>8} {'gap':>7}")
    print("-" * 44)
    rows = []
    for s0 in candidates:
        t0 = time.time(); h0 = best_H0_from(s0, n, m); star = V[idx[s0]]
        gap = 1 - h0 / star; rows.append((gap, s0, star, h0))
        print(f"{str(s0):>14} | {star:>8.2f} {h0:>8.2f} {gap:>7.3f}   ({time.time()-t0:.0f}s)", flush=True)
    rows.sort(reverse=True)
    print(f"\nhardest start: {rows[0][1]}  gap={rows[0][0]:.3f}  (vs all-collided {[r for r in rows if r[1]==(0,0,0,0)][0][0]:.3f})")
