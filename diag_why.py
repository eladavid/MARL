"""Why do 21/27 inits reach the optimum at H=0? Diagnostic.

Uses a precomputed 729-entry Phi lookup table (fast). For a chosen init:
 - roll out the unconstrained joint optimum (value iteration) -> opt trajectory,
 - find the best raw-AD (H=0) joint policy and its trajectory,
 - for the OPTIMAL trajectory, check per agent whether any local state is used
   with >1 distinct action (the conflict that raw AD cannot represent but H=1 can).
"""
import sys, os, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from congestion_game.reward_functions import make_potential_func
from claude_parallelized.parallel_simulation import find_joint_optimum

N, S, A, T, GAMMA = 3, 3, 3, 16, 0.99
_Phi = make_potential_func(S)
# 729-entry lookup: PHI[(state_tuple, action_tuple)] = scalar potential
PHI = {(s, a): float(_Phi(torch.tensor(s), torch.tensor(a)))
       for s in itertools.product(range(S), repeat=N)
       for a in itertools.product(range(A), repeat=N)}

def rollout(init, policy_fn):
    s = tuple(init); tot = 0.0; disc = 1.0; traj = [s]
    for _ in range(T):
        a = policy_fn(s); tot += disc * PHI[(s, a)]; disc *= GAMMA
        s = a; traj.append(s)
    return tot, traj

opt_policy = find_joint_optimum(N, S, A, _Phi, gamma=GAMMA)
opt_fn = lambda s: tuple(int(x) for x in opt_policy[s])
agent_maps = list(itertools.product(range(A), repeat=S))

def best_ad0(init):
    best_v, best_m = -1e9, None
    for m in itertools.product(agent_maps, repeat=N):
        fn = (lambda mm: (lambda s: tuple(mm[i][s[i]] for i in range(N))))(m)
        v, _ = rollout(init, fn)
        if v > best_v: best_v, best_m = v, m
    return best_v, best_m

def conflicts_in_traj(traj):
    """Per agent: local_state -> set of actions taken (action = next local state)."""
    out = {}
    for i in range(N):
        seen = {}
        for t in range(len(traj) - 1):
            s_i, a_i = traj[t][i], traj[t + 1][i]   # action == next state (s'=a)
            seen.setdefault(s_i, set()).add(a_i)
        conf = {s: acts for s, acts in seen.items() if len(acts) > 1}
        if conf: out[i] = conf
    return out

for init in [(0, 0, 0), (1, 1, 1)]:
    print(f"\n================ init {init} ================")
    opt_v, opt_traj = rollout(init, opt_fn)
    ad_v, ad_m = best_ad0(init)
    print(f"optimum   disc-potential = {opt_v:.2f}")
    print(f"best AD^0 disc-potential = {ad_v:.2f}   (ratio {ad_v/opt_v:.3f})")
    print(f"optimal trajectory  : {opt_traj[:8]} ...")
    _, ad_traj = rollout(init, (lambda mm: (lambda s: tuple(mm[i][s[i]] for i in range(N))))(ad_m))
    print(f"best-AD^0 trajectory: {ad_traj[:8]} ...")
    conf = conflicts_in_traj(opt_traj)
    if conf:
        print("CONFLICT in optimal traj (raw AD cannot represent; needs buffer):")
        for i, c in conf.items():
            for s_i, acts in c.items():
                print(f"   agent {i}: local state {s_i} must take actions {sorted(acts)} at different times")
    else:
        print("no per-agent local-state action conflict -> optimal traj IS raw-AD representable")
