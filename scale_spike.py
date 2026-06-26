"""Scaling spike (Phase 1): parameterized rotation-coverage congestion game.

Generalizes the 3x3x3 toy to n agents / m=n slots, preserving the structure:
  - states = actions = {0..m-1}; deterministic decoupled dynamics s'=a; fixed start.
  - g(a): big coordination BONUS if the action profile is a PERMUTATION (full coverage,
    no collision); otherwise the soft congestion penalty 1 - sum((count/n)^2).
  - u_i(s,a) = C * 1[a != s]: move-rewarding (selfish), so STAYING is cheap on g but
    loses u, and the optimum must keep moving.

Phenomenology (verified below):
  - Optimum = a ROTATION through permutations: g BONUS + every agent moves every step,
    Phi*/step = BONUS + n*C.  Trap = a STATIC permutation: g BONUS but no u, Phi/step = BONUS.
  - From a COLLIDED fixed start s0=(0,..,0), reaching the rotation needs an initial
    symmetry-break that conflicts with the steady-state action at state 0  ->  memory gap.
  - Phi* is CLOSED FORM, so we don't need value iteration at large n.

This phase: confirm Phi* (joint VI) == rotation closed form, the optimum is cyclic, the
trap is below it, and tabulate where brute force (VI / H=0 enumeration) explodes.
"""
import time, itertools, numpy as np

BONUS = 10.0     # coordination bonus for a full-coverage permutation (matches 3x3x3 scale)
C     = 3.0      # per-agent move reward (matches 3x3x3 u scale)
GAMMA = 1.0   # undiscounted finite-horizon, matches the paper
T     = 16


def phi(state, action, n):
    """Potential Phi(s,a) = g(a) + sum_i u_i(s_i,a_i) for one joint (state, action)."""
    a = np.asarray(action)
    distinct = len(set(a.tolist())) == n           # permutation <=> all slots distinct
    if distinct:
        g = BONUS
    else:
        counts = np.bincount(a, minlength=n)
        g = 1.0 - np.sum((counts / n) ** 2)
    u = C * np.sum(np.asarray(state) != a)         # move reward, summed over agents
    return g + u


def joint_vi(n, m, T=T, gamma=GAMMA):
    """Finite-horizon joint value iteration. Dynamics s'=a (deterministic). Returns
    (Phi* from collided start, optimal trajectory). Cost ~ m^n * m^n * T."""
    states = list(itertools.product(range(m), repeat=n))
    actions = states                                # same set
    idx = {s: k for k, s in enumerate(states)}
    # precompute phi(s,a) table
    Phi = np.zeros((len(states), len(actions)))
    for si, s in enumerate(states):
        for ai, a in enumerate(actions):
            Phi[si, ai] = phi(s, a, n)
    V = np.zeros(len(states))                       # V_T = 0
    argA = np.zeros((T, len(states)), dtype=int)
    for t in range(T - 1, -1, -1):
        Q = Phi + gamma * V[np.newaxis, :]          # next state = a, so V indexed by action
        argA[t] = Q.argmax(1)
        V = Q.max(1)
    s0 = tuple([0] * n)                             # COLLIDED start
    # roll out optimal trajectory
    traj = []
    s = s0
    for t in range(T):
        a = actions[argA[t, idx[s]]]
        traj.append(a)
        s = a
    return V[idx[s0]], traj


def rollout_value(traj_actions, s0, gamma=GAMMA):
    """Discounted Phi of a fixed action sequence from s0 (dynamics s'=a)."""
    s = s0; V = 0.0
    for t, a in enumerate(traj_actions):
        V += (gamma ** t) * phi(s, a, len(s0))
        s = tuple(a)
    return V


def closed_form_optimum(n, T=T, gamma=GAMMA):
    """Rotation: 1 transient step (collided->permutation), then BONUS + n*C every step."""
    s0 = tuple([0] * n)
    # transient: from all-0, go to the identity permutation (0,1,..,n-1)
    perm = tuple(range(n))
    traj = [perm]                                   # step 0: everyone moves to a distinct slot
    cur = perm
    for t in range(1, T):
        cur = tuple((x + 1) % n for x in cur)       # uniform +1 rotation keeps it a permutation
        traj.append(cur)
    return rollout_value(traj, s0, gamma), traj


def best_static_value(n, m, T=T, gamma=GAMMA):
    """Best fixed action a* held forever from collided start (the trap family)."""
    s0 = tuple([0] * n)
    best = -1e9
    for a in itertools.product(range(m), repeat=n):
        V = rollout_value([a] * T, s0, gamma)
        best = max(best, V)
    return best


def enumerate_H0(n, m, T=T, gamma=GAMMA):
    """Exhaustive over all deterministic MEMORYLESS AD policies (pi_i: slot->slot).
    Cost (m^m)^n. Returns best discounted Phi reachable and the optimum ratio."""
    s0 = tuple([0] * n)
    per_agent = list(itertools.product(range(m), repeat=m))   # each agent's lookup table
    best = -1e9
    for profile in itertools.product(per_agent, repeat=n):    # joint memoryless policy
        s = s0; V = 0.0
        for t in range(T):
            a = tuple(profile[i][s[i]] for i in range(n))
            V += (gamma ** t) * phi(s, a, n)
            s = a
        best = max(best, V)
    return best


if __name__ == "__main__":
    print(f"game: m=n slots, BONUS={BONUS}, C={C}, T={T}, gamma={GAMMA}\n")
    print(f"{'n':>2} | {'Phi*(VI)':>10} {'closed-form':>12} {'best-static':>12} "
          f"{'gap(stat)':>10} | {'VI cost':>14} {'H0 enum cost':>16}")
    print("-" * 92)
    for n in [3, 4, 5]:
        m = n
        t0 = time.time()
        vi_star, traj = joint_vi(n, m)
        vi_t = time.time() - t0
        cf, _ = closed_form_optimum(n)
        stat = best_static_value(n, m)
        vi_cost = (m ** n) ** 2 * T
        h0_cost = (m ** m) ** n
        print(f"{n:>2} | {vi_star:>10.3f} {cf:>12.3f} {stat:>12.3f} "
              f"{1 - stat/vi_star:>10.3f} | {vi_cost:>14,d} {h0_cost:>16,d}   (VI {vi_t:.2f}s)")
        # show the optimal trajectory is a cycle (period n rotation), not static
        tail = traj[-(n + 1):]
        print(f"     opt tail (should rotate, period {n}): {tail}")
    print()
    # H=0 exhaustive only where tractable
    for n in [3]:
        t0 = time.time()
        h0 = enumerate_H0(n, n)
        vi_star, _ = joint_vi(n, n)
        print(f"H=0 exhaustive  n={n}: best={h0:.3f}  Phi*={vi_star:.3f}  "
              f"ratio={h0/vi_star:.4f}  gap={1-h0/vi_star:.4f}  ({time.time()-t0:.2f}s)")
