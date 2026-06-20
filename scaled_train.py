"""MAC-REINFORCE with a FIXED buffer (distinct pad token = m, not 0), so an agent that
sits on the collided depot slot is representable. Pluggable g/u. Used to re-test the
k x k x k game at the TRUE required buffer (H*=1) and isolate selection from the old
pad-0 artifact."""
import sys, os, time, random as pyrandom
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
torch.set_num_threads(1)   # many tiny per-agent ops -> single thread avoids thread-thrash / unstable timing
from congestion_game.policies import DirectTabularPolicy
from vectorized_train import disc_returns, reinforce_loss

def make_policies(n, m, H, alpha=0.1):
    vocab = [m + 1] * H + [m]            # H pad-capable history slots (0..m) + current state (0..m-1)
    return [DirectTabularPolicy(vocab, m, alpha) for _ in range(n)]

def rollout(policies, init, g_fn, u_fn, m, H, T, B, gamma, greedy=False):
    n = len(policies); PAD = m
    curr = torch.stack([torch.full((B,), int(s), dtype=torch.long) for s in init])   # (n,B)
    buf = torch.full((n, B, H), PAD, dtype=torch.long)
    logps, rews, pots = [], [], []
    for t in range(T):
        step_lp, acts = [], []
        for i in range(n):
            aug = torch.cat([buf[i], curr[i].unsqueeze(1)], dim=1)
            probs = policies[i](aug).view(-1, m)
            if greedy:
                a = torch.argmax(probs, dim=1); step_lp.append(torch.zeros(B))
            else:
                dist = torch.distributions.Categorical(probs)
                a = dist.sample(); step_lp.append(dist.log_prob(a))
            acts.append(a)
        actions = torch.stack(acts, dim=1)
        g = g_fn(actions, n, m); us = [u_fn(curr[i], actions[:, i], m) for i in range(n)]
        rews.append(torch.stack([g + u for u in us], dim=1)); pots.append(g + sum(us))
        logps.append(torch.stack(step_lp, dim=1))
        if H > 0:
            buf = torch.cat([buf[:, :, 1:], curr.unsqueeze(2)], dim=2)
        curr = actions.t().contiguous()
    return torch.stack(logps), torch.stack(rews), torch.stack(pots)

def train(policies, init, g_fn, u_fn, m, H, T, episodes, batch, gamma, lr):
    opts = [torch.optim.SGD(p.parameters(), lr=lr) for p in policies]
    for _ in range(episodes):
        logps, rews, _ = rollout(policies, init, g_fn, u_fn, m, H, T, batch, gamma)
        loss = reinforce_loss(logps, rews, gamma, batch, True)
        for o in opts: o.zero_grad()
        loss.backward()
        for o, p in zip(opts, policies):
            o.step(); p.project_parameters_onto_simplex()

def greedy_phi(policies, init, g_fn, u_fn, m, H, T, gamma):
    _, _, pots = rollout(policies, init, g_fn, u_fn, m, H, T, 1, gamma, greedy=True)
    return float((disc_returns(pots.detach(), gamma)[0]).mean())

def run(n, m, g_fn, u_fn, phistar, H, seeds, episodes, batch, T, gamma, lr, label, alpha=0.1):
    init = tuple([0] * n); ratios, t0 = [], time.time()
    for seed in range(seeds):
        pyrandom.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        pols = make_policies(n, m, H, alpha)
        train(pols, init, g_fn, u_fn, m, H, T, episodes, batch, gamma, lr)
        ratios.append(greedy_phi(pols, init, g_fn, u_fn, m, H, T, gamma) / phistar)
    r = np.array(ratios); dt = time.time() - t0
    print(f"{label} H={H} | reach(>=0.99)={int((r>=0.99).sum())}/{seeds} "
          f"best={r.max():.3f} mean={r.mean():.3f} | {dt/seeds:.1f}s/seed", flush=True)
    return r

if __name__ == "__main__":
    from scale_spike2 import g_gen, u_gen          # k x k x k permutation-bonus game
    from scale_spike import closed_form_optimum
    T, GAMMA = 16, 0.99
    print("k x k x k, FIXED buffer (pad=m). H=0 (gap) vs H=1 (req buffer): reach vs k\n")
    for k in [3, 4, 5, 6]:
        cf, _ = closed_form_optimum(k)
        for H in [0, 1]:
            run(k, k, g_gen, u_gen, cf, H, 6, 2000, 64, T, GAMMA, 1e-3, f"k={k}")
        print()
