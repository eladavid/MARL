"""Drone congestion game (paper section 2.1.1 instantiation), VI reference.

n agents on 5 nodes: R={0,1} reload, M={2,3,4} mission. Fully connected.
Per-agent state s_i=(node, battery), battery b in {0..B}. Action a_i = next node.
Deterministic decoupled dynamics:
  node' = a_i ;  battery' = B if a_i in R else max(0, b-1)   (recharge to full, else drain 1)
Rewards:
  g(a)  = W_cov * (#distinct MISSION nodes occupied)/|M|     (coverage; redundancy wasted)
  u_i   = -C_batt * (1 - battery'_i / B)                     (battery/travel aversion: prefer full)
Phi = g + sum_i u_i. Battery FORCES periodic reload, so continuous coverage needs agents to
take turns reloading -> the optimum is a coordinated cyclic patrol; a static parking policy
drains every battery -> trap. VI feasible to ~n=4 (25^n states); dead at n=5 (the point).
"""
import itertools, time, numpy as np

R = {0}; M = {1, 2, 3, 4}; NODES = 5; B = 4      # single reload bottleneck + 4 missions (>n agents): harder
W_COV, C_BATT, W_CONG = 10.0, 3.0, 4.0
GAMMA, T = 0.99, 16
BATT = B + 1                                   # battery levels 0..B

FROZEN_RELOAD = 0          # at b=0 the frozen (non-parameterized) policy sends the drone to R0

def frozen_action(b, a):
    """Action actually executed: the learnable action a when b>0; at b=0 a FROZEN policy
    overrides it to 'go reload' (R0). Not parameterized, not learned."""
    return a if b > 0 else FROZEN_RELOAD

def step_battery(b, a):                          # standard decoupled dynamics s'=a (a already frozen-resolved)
    return B if a in R else max(0, b - 1)

def g_of(action):
    occ = set(a for a in action if a in M)
    # limited node capacity: penalize agents stacking on the same node (incl. reload stations)
    counts = {}
    for a in action: counts[a] = counts.get(a, 0) + 1
    cong = sum(c - 1 for c in counts.values() if c > 1)
    return W_COV * len(occ) / len(M) - W_CONG * cong

def u_of(b, a):
    bp = step_battery(b, a)
    return -C_BATT * (1 - bp / B)

def phi(state, action, n):
    # resolve the frozen reload at empty battery, then standard g/u on the executed actions
    ea = tuple(frozen_action(state[i][1], action[i]) for i in range(n))
    return g_of(ea) + sum(u_of(state[i][1], ea[i]) for i in range(n))

def all_states(n):
    cell = [(v, b) for v in range(NODES) for b in range(BATT)]   # 25 per agent
    return list(itertools.product(cell, repeat=n))

def _nxt(s, a, n):                               # executed action resolves the frozen reload at b=0
    return tuple((frozen_action(s[i][1], a[i]), step_battery(s[i][1], frozen_action(s[i][1], a[i]))) for i in range(n))

def vi(n, s0):
    states = all_states(n); idx = {s: k for k, s in enumerate(states)}
    actions = list(itertools.product(range(NODES), repeat=n))
    V = np.zeros(len(states)); argA = np.zeros((T, len(states)), dtype=int)
    Phi = np.array([[phi(s, a, n) for a in actions] for s in states])
    NX = np.array([[idx[_nxt(s, a, n)] for a in actions] for s in states])
    for t in range(T - 1, -1, -1):
        Q = Phi + GAMMA * V[NX]; argA[t] = Q.argmax(1); V = Q.max(1)
    s = s0; traj = []
    for t in range(T):
        a = tuple(frozen_action(s[i][1], actions[argA[t, idx[s]]][i]) for i in range(n))  # report executed action
        traj.append(a); s = _nxt(s, actions[argA[t, idx[s]]], n)
    return V[idx[s0]], traj

def rollout_value(policy_actions, s0, n):
    s = s0; V = 0.0
    for t, a in enumerate(policy_actions):
        V += GAMMA ** t * phi(s, a, n)
        s = _nxt(s, a, n)
    return V

def best_static(n, s0):
    """best single fixed joint action held forever (the parking trap family)."""
    best = -1e9; ba = None
    for a in itertools.product(range(NODES), repeat=n):
        v = rollout_value([a] * T, s0, n)
        if v > best: best, ba = v, a
    return best, ba

if __name__ == "__main__":
    n = 3
    s0 = tuple([(0, B)] * n)                    # all start at reload R0, full battery
    t0 = time.time(); star, traj = vi(n, s0); dt = time.time() - t0
    stat, ba = best_static(n, s0)
    print(f"DRONE GAME n={n}: |S_i|={NODES*BATT} joint states={(NODES*BATT)**n}  "
          f"W_cov={W_COV} C_batt={C_BATT} B={B} T={T}")
    print(f"  Phi*(VI)={star:.2f}  best-static={stat:.2f} (action {ba})  "
          f"trap_ratio={stat/star:.3f}  (VI {dt:.1f}s)\n")
    print("  optimal trajectory (node,batt) per agent  ->  shows reload rotation + coverage:")
    def fmt(s): return " ".join(f"{'R' if v in R else 'M'}{v}:{b}" for v, b in s)
    s = s0
    for t, a in enumerate(traj):
        cov = len(set(x for x in a if x in M))
        nl = [('R' if x in R else 'M', x) for x in a]
        print(f"   t={t:2d}  act={a}  cover={cov}/3  " +
              ("reloading: " + ",".join(f"ag{i}" for i in range(n) if a[i] in R) if any(x in R for x in a) else "all on mission"))
        s = tuple((a[i], step_battery(s[i][1], a[i])) for i in range(n))
    # detect cycle period in the tail
    tail = traj[6:]
    per = next((p for p in range(1, 6) if all(tail[k] == tail[k+p] for k in range(len(tail)-p-1))), None)
    print(f"\n  tail cycle period = {per}  (1 = static; >1 = coordinated rotating patrol)")
