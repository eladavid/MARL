"""Fig 3b data: hopping concentration nu^beta on the graded game (rho=0.7, k=4).
Builds a pool WITH per-candidate (G, U_vec, Phi) stats, runs the local-unanimous-vote
hopping chain over a beta grid, saves sim_figs/hopping_data.pkl."""
import sys, os, pickle, random as pyrandom
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from scaled_vec import init_P, train, rollout, _strides
from graded_game import make_g_torch, closed_form
from scale_spike2 import u_gen
T, GAMMA, RHO = 16, 0.99, 0.7
g_fn = make_g_torch(RHO)

def stat(P, n, m, H, alpha, strides_t):
    _, rews, pots = rollout(P, tuple([0]*n), g_fn, u_gen, m, H, T, 1, GAMMA, alpha, strides_t, greedy=True)
    w = GAMMA ** torch.arange(T)
    rt = rews.squeeze(-1); pt = pots.squeeze(-1)
    g_t = (rt.sum(1) - pt) / (n - 1); u_t = rt - g_t.unsqueeze(1)
    G = float((g_t * w).sum()); U = (u_t * w.unsqueeze(1)).sum(0).numpy()
    return (G, U, G + float(U.sum()))

def build(k, Q, episodes=5000, batch=32, H=1, alpha=0.1, lr=1e-3):
    m = k; strides_t = torch.tensor(_strides([m+1]*H + [m])); init = tuple([0]*k); pool = []
    for c in range(Q):
        pyrandom.seed(c); np.random.seed(c); torch.manual_seed(c)
        P = init_P(k, m, H)
        train(P, init, g_fn, u_gen, m, H, T, episodes, batch, GAMMA, lr, alpha, strides_t)
        pool.append(stat(P, k, m, H, alpha, strides_t))
    return pool

def accepts(sb, sa, beta, n, rng):
    Gb, Ub, _ = sb; Ga, Ua, _ = sa
    for i in range(n):
        d = (Gb - Ga) / n + (Ub[i] - Ua[i])
        if rng.random() >= min(1.0, np.exp(np.clip(d / beta, -50, 50))):   # clip avoids overflow
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
    k = 5; opt = closed_form(k, RHO)        # k=5: trap-heavy, dramatic concentration
    pool = build(k, 150)
    ratios = np.array([s[2] / opt for s in pool])
    betas = [0.3, 0.15, 0.08, 0.04, 0.02, 0.01]
    nu = {b: float(np.mean([chain(pool, b, k, opt, seed=s) for s in range(6)])) for b in betas}
    # pool-baseline (no selection) = fraction optimal in pool
    base = float((ratios >= 0.99).mean())
    pickle.dump({"k": k, "betas": betas, "nu": nu, "pool_optfrac": base,
                 "pool_ratios": ratios.tolist()}, open("sim_figs/hopping_data.pkl", "wb"))
    print(f"hopping k={k}: pool_optfrac={base:.2f}  nu^beta={ {b: round(v,3) for b,v in nu.items()} }")
