"""Vectorized MAC-REINFORCE: all n agents in ONE batched tensor (no per-agent Python
loop), single batched simplex projection, single backward. Faithful to scaled_train.py
(direct tabular params, alpha-mix, fixed pad=m buffer, reward-to-go + mean baseline).
Kills the ~k^4 per-step blowup -> near-flat scaling in n."""
import sys, os, time, random as pyrandom
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
torch.set_num_threads(1)
from congestion_game.policies import project_onto_simplex
from vectorized_train import disc_returns

def _strides(V):
    s = [1] * len(V)
    for k in range(len(V) - 2, -1, -1):
        s[k] = s[k + 1] * V[k + 1]
    return s

def init_P(n, m, H):
    S = (m + 1) ** H * m                                   # flat augmented-state count
    flat = torch.distributions.Dirichlet(torch.ones(m)).sample((n, S))   # (n,S,m) rows on simplex
    return torch.nn.Parameter(flat)

def _probs(P, aug, strides_t, m, alpha):
    n, B = aug.shape[0], aug.shape[1]
    flat_idx = (aug * strides_t).sum(-1)                   # (n,B)
    p = P[torch.arange(n).unsqueeze(1), flat_idx]          # (n,B,m), differentiable gather
    p = torch.clamp(p, min=0.0); p = p / p.sum(-1, keepdim=True)
    return (1 - alpha) * p + alpha / m

def rollout(P, init, g_fn, u_fn, m, H, T, B, gamma, alpha, strides_t, greedy=False):
    n = len(init); PAD = m
    curr = torch.stack([torch.full((B,), int(s), dtype=torch.long) for s in init])    # (n,B)
    buf = torch.full((n, B, H), PAD, dtype=torch.long)
    logps, rews, pots = [], [], []
    for t in range(T):
        aug = torch.cat([buf, curr.unsqueeze(2)], dim=2)                   # (n,B,H+1)
        probs = _probs(P, aug, strides_t, m, alpha)                        # (n,B,m)
        if greedy:
            a = probs.argmax(-1); lp = torch.zeros(n, B)
        else:
            dist = torch.distributions.Categorical(probs); a = dist.sample(); lp = dist.log_prob(a)
        actions_Bn = a.t().contiguous()                                    # (B,n)
        g = g_fn(actions_Bn, n, m)                                         # (B,)
        u = torch.stack([u_fn(curr[i], a[i], m) for i in range(n)])        # (n,B)
        rews.append(g.unsqueeze(0) + u)                                    # (n,B): per-agent r_i = g + u_i
        pots.append(g + u.sum(0))                                          # (B,):  potential Phi = g + sum u_i
        logps.append(lp)
        if H > 0:
            buf = torch.cat([buf[:, :, 1:], curr.unsqueeze(2)], dim=2)
        curr = a
    return torch.stack(logps), torch.stack(rews), torch.stack(pots)       # (T,n,B),(T,n,B),(T,B)

def train(P, init, g_fn, u_fn, m, H, T, episodes, batch, gamma, lr, alpha, strides_t):
    opt = torch.optim.SGD([P], lr=lr)
    n, S = P.shape[0], P.shape[1]
    for _ in range(episodes):
        logps, rews, _ = rollout(P, init, g_fn, u_fn, m, H, T, batch, gamma, alpha, strides_t)
        rets = disc_returns(rews.detach(), gamma)                         # (T,n,B)
        adv = rets - rets.mean(dim=2, keepdim=True)                       # baseline = mean over batch
        loss = -(logps * adv).sum() / batch
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            P.copy_(project_onto_simplex(P.view(n * S, m)).view(n, S, m))

def greedy_phi(P, init, g_fn, u_fn, m, H, T, gamma, alpha, strides_t):
    _, _, pots = rollout(P, init, g_fn, u_fn, m, H, T, 1, gamma, alpha, strides_t, greedy=True)
    return float(disc_returns(pots.detach(), gamma)[0].mean())           # pots already = Phi (T,B)

def run(n, m, g_fn, u_fn, phistar, H, seeds, episodes, batch, T, gamma, lr, label, alpha=0.1):
    init = tuple([0] * n); V = [m + 1] * H + [m]; strides_t = torch.tensor(_strides(V))
    ratios, t0 = [], time.time()
    for seed in range(seeds):
        pyrandom.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        P = init_P(n, m, H)
        train(P, init, g_fn, u_fn, m, H, T, episodes, batch, gamma, lr, alpha, strides_t)
        ratios.append(greedy_phi(P, init, g_fn, u_fn, m, H, T, gamma, alpha, strides_t) / phistar)
    r = np.array(ratios); dt = time.time() - t0
    print(f"{label} H={H} | reach(>=0.99)={int((r>=0.99).sum())}/{seeds} "
          f"best={r.max():.3f} mean={r.mean():.3f} | {dt/seeds:.1f}s/seed", flush=True)
    return r

if __name__ == "__main__":
    from scale_spike2 import g_gen, u_gen
    from scale_spike import closed_form_optimum
    T, GAMMA = 16, 1.0
    print("VECTORIZED correctness+timing vs loop version (expect k=3 H=1 -> 1.0):\n")
    for k in [3, 4, 5, 6]:
        cf, _ = closed_form_optimum(k)
        run(k, k, g_gen, u_gen, cf, 1, 6, 2000, 64, T, GAMMA, 1e-3, f"k={k}")
