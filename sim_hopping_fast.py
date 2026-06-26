"""Fig 3b (fast): hopping concentration nu^beta at k=5, graded rho=0.7.
Builds the pool with the FAST candidate-vectorized trainer (scaled_pool) and extracts
per-candidate (G, U_vec, Phi) stats from one batched greedy rollout, then runs the
local-unanimous-vote hopping chains over a beta grid. Saves sim_figs/hopping_data.pkl."""
import sys, os, pickle, time, random as pyrandom
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
torch.set_num_threads(1)
from congestion_game.policies import project_onto_simplex
from vectorized_train import disc_returns
from scaled_pool import rollout as prollout, _strides
import scaled_pool
from graded_game import make_g_torch, closed_form
T, GAMMA, RHO = 16, 0.99, 0.7
scaled_pool.g_gen = make_g_torch(RHO)

def build_pool_stats(k, Q, episodes=5000, batch=32, H=1, alpha=0.1, lr=1e-3, seed=0):
    """Train Q candidates at once; return list of (G, U_vec, Phi) from greedy rollout."""
    m = k; strides_t = torch.tensor(_strides([m + 1] * H + [m]))
    S = (m + 1) ** H * m
    torch.manual_seed(seed)
    P = torch.nn.Parameter(torch.distributions.Dirichlet(torch.ones(m)).sample((Q, k, S)))
    optim = torch.optim.SGD([P], lr=lr)
    for _ in range(episodes):
        logps, rews, _ = prollout(P, Q, k, m, H, batch, alpha, strides_t)
        rets = disc_returns(rews.detach(), GAMMA); adv = rets - rets.mean(dim=3, keepdim=True)
        loss = -(logps * adv).sum() / batch
        optim.zero_grad(); loss.backward(); optim.step()
        with torch.no_grad():
            P.copy_(project_onto_simplex(P.view(Q * k * S, m)).view(Q, k, S, m))
    with torch.no_grad():
        _, rews, pots = prollout(P, Q, k, m, H, 1, alpha, strides_t, greedy=True)  # (T,Q,n,1),(T,Q,1)
        w = (GAMMA ** torch.arange(T)).view(T, 1)
        rt = rews.squeeze(-1); pt = pots.squeeze(-1)                # (T,Q,n),(T,Q)
        g_t = (rt.sum(2) - pt) / (k - 1)                           # (T,Q)
        u_t = rt - g_t.unsqueeze(2)                                # (T,Q,n)
        G = (g_t * w).sum(0).numpy()                               # (Q,)
        U = (u_t * w.unsqueeze(2)).sum(0).numpy()                  # (Q,n)
    return [(float(G[q]), U[q], float(G[q] + U[q].sum())) for q in range(Q)]

def accepts(sb, sa, beta, n, rng):
    Gb, Ub, _ = sb; Ga, Ua, _ = sa
    for i in range(n):
        d = (Gb - Ga) / n + (Ub[i] - Ua[i])
        if rng.random() >= min(1.0, np.exp(np.clip(d / beta, -50, 50))):
            return False
    return True

def chain(pool, beta, n, opt, epochs=4000, burn=400, seed=0):
    rng = pyrandom.Random(seed); inc = min(pool, key=lambda s: s[2]); at = 0
    for e in range(epochs):
        c = pool[rng.randrange(len(pool))]
        if accepts(c, inc, beta, n, rng): inc = c
        if e >= burn and inc[2] / opt >= 0.99: at += 1
    return at / (epochs - burn)

if __name__ == "__main__":
    k = 5; opt = closed_form(k, RHO); t0 = time.time()
    pool = build_pool_stats(k, 200)
    ratios = np.array([s[2] / opt for s in pool])
    betas = [0.3, 0.15, 0.08, 0.04, 0.02, 0.01]
    nu = {b: float(np.mean([chain(pool, b, k, opt, seed=s) for s in range(6)])) for b in betas}
    base = float((ratios >= 0.99).mean())
    pickle.dump({"k": k, "betas": betas, "nu": nu, "pool_optfrac": base,
                 "pool_ratios": ratios.tolist()}, open("sim_figs/hopping_data.pkl", "wb"))
    print(f"hopping k={k} ({time.time()-t0:.0f}s): pool_optfrac={base:.2f}  "
          f"nu={ {b: round(v,3) for b,v in nu.items()} }")
