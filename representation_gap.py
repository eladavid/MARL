"""Step 1 (Fig 3 foundation): representation gap of raw-AD (H=0).

Training-free. Enumerates ALL deterministic per-agent raw-AD policies (each agent
maps its own state -> action; 3^3=27 per agent, 27^3 joint) and computes the best
achievable discounted episode potential from each initial joint state, vs the
unconstrained joint optimum (value iteration). If best-AD^(0) < optimum, raw AD is
provably gap-limited -> motivates the history buffer.
"""
import sys, os, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from congestion_game.reward_functions import make_potential_func
from claude_parallelized.parallel_simulation import find_joint_optimum  # value iteration

N, S, A, T, GAMMA = 3, 3, 3, 16, 0.99
Phi = make_potential_func(S)  # potential(joint_state_tensor, joint_action_tensor) = g + sum u_i

def disc_potential_rollout(init, joint_policy_fn):
    """Deterministic rollout from init; joint_policy_fn(state_tuple)->action_tuple."""
    s = tuple(init); tot = 0.0; disc = 1.0
    for _ in range(T):
        a = joint_policy_fn(s)
        r = float(Phi(torch.tensor(s), torch.tensor(a)))
        tot += disc * r; disc *= GAMMA
        s = a                      # deterministic transition s' = a
    return tot

# unconstrained joint optimum (joint-state -> joint-action) via value iteration
opt_policy = find_joint_optimum(N, S, A, Phi, gamma=GAMMA)
opt_fn = lambda s: tuple(int(x) for x in opt_policy[s])

# all per-agent raw-AD maps: tuple m of length S, m[state]=action
agent_maps = list(itertools.product(range(A), repeat=S))   # 27 maps
print(f"per-agent AD^(0) maps: {len(agent_maps)} | joint: {len(agent_maps)**N}", flush=True)

inits = list(itertools.product(range(S), repeat=N))        # all 27 initial joint states
print(f"{'init':>10} {'opt':>8} {'bestAD0':>8} {'gap':>7} {'ratio':>6}", flush=True)
rows = []
for init in inits:
    opt_val = disc_potential_rollout(init, opt_fn)
    best_ad0 = -1e9
    for m in itertools.product(agent_maps, repeat=N):      # joint AD^(0) policy
        fn = (lambda mm: (lambda s: tuple(mm[i][s[i]] for i in range(N))))(m)
        v = disc_potential_rollout(init, fn)
        if v > best_ad0:
            best_ad0 = v
    gap = opt_val - best_ad0
    ratio = best_ad0 / opt_val if opt_val else float('nan')
    rows.append((init, opt_val, best_ad0, gap, ratio))
    print(f"{str(init):>10} {opt_val:8.2f} {best_ad0:8.2f} {gap:7.2f} {ratio:6.3f}", flush=True)

import statistics
gaps = [r[3] for r in rows]; ratios = [r[4] for r in rows]
n_gap = sum(1 for g in gaps if g > 1e-6)
print("\n=== SUMMARY (AD^(0) vs joint optimum) ===", flush=True)
print(f"initial states with a strict gap: {n_gap}/{len(rows)}")
print(f"mean ratio best-AD^(0)/optimum   : {statistics.mean(ratios):.3f}  (min {min(ratios):.3f}, max {max(ratios):.3f})")
print(f"mean gap                         : {statistics.mean(gaps):.2f}  (max {max(gaps):.2f})")
