"""Vectorized MAC-REINFORCE for the toy congestion game.

Runs all `batch_size` episode rollouts as ONE batched tensor op instead of a Python
for-loop, for a large speedup. Same REINFORCE math (reward-to-go, mean baseline,
SGD, simplex projection), same DirectTabularPolicy (incl. its alpha-greedy mixing),
same deterministic dynamics s'=a, same g/u rewards. Verified against the original
sequential `train()` in test_vectorized.py.

Only models the structured congestion game (g_func + make_u_i). H>=1 supported via a
pad-0 history buffer matching EpisodicAgent.get_augmented_state.
"""
import torch

# coordinated optima of g_func (see congestion_game/reward_functions.py)
_OPT = (torch.tensor([0, 1, 2]), torch.tensor([0, 2, 1]))


def g_batched(actions, N, A):
    """actions: (B,N) long -> (B,) float. Matches reward_functions.g_func."""
    is_opt = (actions == _OPT[0]).all(1) | (actions == _OPT[1]).all(1)
    counts = torch.nn.functional.one_hot(actions, A).sum(1).float()       # (B,A)
    cong = 1.0 - ((counts / N) ** 2).sum(1)                                # (B,)
    return torch.where(is_opt, torch.full_like(cong, 10.0), cong)


def u_batched(s, a, S):
    """s,a: (B,) long -> (B,) float. Matches make_u_i(S): s==0 rewards a near 0, else far."""
    s = s.float(); a = a.float(); d = (s - a).abs() / S
    return torch.where(s == 0, 3.0 * (1.0 - d), 3.0 * d)


def disc_returns(x, gamma):
    """Reverse discounted reward-to-go along dim 0. x:(T,...) -> (T,...). Matches utils."""
    out = torch.zeros_like(x); R = torch.zeros_like(x[0])
    for t in reversed(range(x.shape[0])):
        R = x[t] + gamma * R; out[t] = R
    return out


def vec_rollout(policies, init_states, S, A, H, T, B, fixed_actions=None):
    """Batched rollout of B episodes. Returns logps (T,B,N) [grad], rews (T,B,N), pots (T,B).
    If fixed_actions (T,B,N) given, replay them (for tests) instead of sampling."""
    N = len(policies)
    curr = torch.stack([torch.full((B,), int(s), dtype=torch.long) for s in init_states])  # (N,B)
    buf = torch.zeros(N, B, H, dtype=torch.long)            # pad-0 history (matches get_augmented_state)
    logps, rews, pots = [], [], []
    for t in range(T):
        step_lp, acts = [], []
        for i in range(N):
            aug = torch.cat([buf[i], curr[i].unsqueeze(1)], dim=1)         # (B, H+1)
            probs = policies[i](aug)                                       # (B, A), incl. exploration mix
            dist = torch.distributions.Categorical(probs)
            a = dist.sample() if fixed_actions is None else fixed_actions[t, :, i]
            step_lp.append(dist.log_prob(a)); acts.append(a)
        actions = torch.stack(acts, dim=1)                                 # (B, N)
        g = g_batched(actions, N, A)                                       # (B,)
        us = [u_batched(curr[i], actions[:, i], S) for i in range(N)]      # each (B,)
        rews.append(torch.stack([g + u for u in us], dim=1))               # (B, N)
        pots.append(g + sum(us))                                           # (B,)
        logps.append(torch.stack(step_lp, dim=1))                          # (B, N)
        if H > 0:
            buf = torch.cat([buf[:, :, 1:], curr.unsqueeze(2)], dim=2)     # push curr
        curr = actions.t().contiguous()                                    # (N,B): s' = a
    return torch.stack(logps), torch.stack(rews), torch.stack(pots)


def reinforce_loss(logps, rews, gamma, batch_size, use_baseline):
    """Total REINFORCE loss = sum_i (1/B) sum_{b,t} -logp_i (R_i - baseline_i).
    logps:(T,B,N) grad, rews:(T,B,N). Matches the original train() loss exactly."""
    rets = disc_returns(rews.detach(), gamma)                              # (T,B,N) reward-to-go
    adv = rets - rets.mean(dim=1, keepdim=True) if use_baseline else rets  # baseline = mean over batch (per t, per agent)
    return -(logps * adv).sum() / batch_size


def train_vectorized(policies, init_states, num_episodes, batch_size,
                     S=3, A=3, H=1, T=16, gamma=1.0, lr=1e-3, use_baseline=True):
    """In-place MAC-REINFORCE on a list of DirectTabularPolicy. Returns per-episode
    discounted potential (mean over batch) — same quantity as the original train()."""
    from congestion_game.policies import DirectTabularPolicy
    opts = [torch.optim.SGD(p.parameters(), lr=lr) for p in policies]
    ep_pot = []
    for _ in range(num_episodes):
        logps, rews, pots = vec_rollout(policies, init_states, S, A, H, T, batch_size)
        loss = reinforce_loss(logps, rews, gamma, batch_size, use_baseline)
        for o in opts: o.zero_grad()
        loss.backward()
        for o, p in zip(opts, policies):
            o.step()
            if isinstance(p, DirectTabularPolicy):
                p.project_parameters_onto_simplex()
        ep_pot.append(disc_returns(pots.detach(), gamma)[0].mean().item())
    return ep_pot
