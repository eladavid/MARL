"""Scaling spike (congestion variant): n > |S| pure-congestion game, |S| fixed.

n agents, m slots, n >= m. states = actions = {0..m-1}; s'=a; collided start (0..0).
  g(a) = W * (1 - sum_slot (count/n)^2)     # pure congestion, max at balanced load
  u_i(s,a) = C * 1[a != s]                   # move-rewarding (selfish)
Optimum = BALANCED ROTATION (hold the even split, everyone shifts +1): keeps g maximal
AND every agent moves -> Phi*/step = W*g_balanced + n*C, closed form. We cross-check with
joint VI at small n, confirm the gap (memoryless can't reach it) and the trap, and show
where brute force explodes -- while the PER-AGENT table stays [m]^(H+1), fixed in n.
"""
import time, itertools, numpy as np

W     = 10.0     # congestion weight (tune so coordination matters vs. the move reward)
C     = 3.0
GAMMA = 0.99
T     = 16


def g_balanced(n, m):
    """Best (most even) congestion value: split n over m as evenly as possible."""
    q, r = divmod(n, m)
    counts = np.array([q + 1] * r + [q] * (m - r))
    return W * (1.0 - np.sum((counts / n) ** 2))


def phi(state, action, n, m):
    a = np.asarray(action)
    counts = np.bincount(a, minlength=m)
    g = W * (1.0 - np.sum((counts / n) ** 2))
    u = C * np.sum(np.asarray(state) != a)
    return g + u


def joint_vi(n, m, T=T, gamma=GAMMA):
    states = list(itertools.product(range(m), repeat=n))
    idx = {s: k for k, s in enumerate(states)}
    Phi = np.array([[phi(s, a, n, m) for a in states] for s in states])
    V = np.zeros(len(states)); argA = np.zeros((T, len(states)), dtype=int)
    for t in range(T - 1, -1, -1):
        Q = Phi + gamma * V[np.newaxis, :]
        argA[t] = Q.argmax(1); V = Q.max(1)
    s0 = tuple([0] * n); s = s0; traj = []
    for t in range(T):
        a = states[argA[t, idx[s]]]; traj.append(a); s = a
    return V[idx[s0]], traj


def rollout_value(traj, s0, n, m, gamma=GAMMA):
    s = s0; V = 0.0
    for t, a in enumerate(traj):
        V += (gamma ** t) * phi(s, a, n, m); s = tuple(a)
    return V


def closed_form_optimum(n, m, T=T, gamma=GAMMA):
    """Balanced assignment (agent i -> slot i%m) then uniform +1 rotation."""
    s0 = tuple([0] * n)
    cur = tuple(i % m for i in range(n)); traj = [cur]
    for t in range(1, T):
        cur = tuple((x + 1) % m for x in cur); traj.append(cur)
    return rollout_value(traj, s0, n, m, gamma), traj


def best_static_value(n, m, T=T, gamma=GAMMA):
    s0 = tuple([0] * n); best = -1e9
    for a in itertools.product(range(m), repeat=n):
        best = max(best, rollout_value([a] * T, s0, n, m, gamma))
    return best


def enumerate_H0(n, m, T=T, gamma=GAMMA):
    s0 = tuple([0] * n); per_agent = list(itertools.product(range(m), repeat=m)); best = -1e9
    for profile in itertools.product(per_agent, repeat=n):
        s = s0; V = 0.0
        for t in range(T):
            a = tuple(profile[i][s[i]] for i in range(n)); V += (gamma ** t) * phi(s, a, n, m); s = a
        best = max(best, V)
    return best


if __name__ == "__main__":
    m = 3
    print(f"PURE CONGESTION game, |S|=m={m} FIXED, W={W}, C={C}, T={T}\n")
    print(f"{'n':>2} | {'Phi*(VI)':>10} {'closed':>9} {'static':>8} {'gap_stat':>9} | "
          f"{'g_bal':>6} {'VI cost':>13} {'per-agent tbl':>13}")
    print("-" * 92)
    for n in [3, 4, 5, 6, 7]:
        cf, _ = closed_form_optimum(n, m)
        gb = g_balanced(n, m)
        per_tbl = m ** 2 * m          # [m]^(H+1) with H=1, times A=m
        vi_cost = (m ** n) ** 2 * T
        vi_star = None
        if m ** n <= 4000:            # VI only where the Phi table fits
            t0 = time.time(); vi_star, traj = joint_vi(n, m); vt = time.time() - t0
            stat = best_static_value(n, m)
            tail = traj[-3:]
            print(f"{n:>2} | {vi_star:>10.3f} {cf:>9.3f} {stat:>8.3f} {1-stat/vi_star:>9.3f} | "
                  f"{gb:>6.3f} {vi_cost:>13,d} {per_tbl:>13,d}   (VI {vt:.1f}s, opt tail {tail})")
        else:
            print(f"{n:>2} | {'--':>10} {cf:>9.3f} {'--':>8} {'--':>9} | "
                  f"{gb:>6.3f} {vi_cost:>13,d} {per_tbl:>13,d}   (VI infeasible: {m**n} states)")
    print()
    for n in [4, 5]:                  # gap: memoryless ceiling where enumeration is tractable
        if (m ** m) ** n <= 6e7:
            t0 = time.time(); h0 = enumerate_H0(n, m); vs, _ = joint_vi(n, m)
            print(f"H=0 exhaustive n={n},m={m}: best={h0:.3f} Phi*={vs:.3f} "
                  f"ratio={h0/vs:.4f} gap={1-h0/vs:.4f} ({time.time()-t0:.1f}s)")
