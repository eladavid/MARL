"""Hopping (Algorithm 2) on the scaled k x k x k game, vectorized.
Pool = stationary policies from many local searches (random-init MAC-REINFORCE), each
recorded as a stat (G, U_vec, Phi). Hopping chain replays the pool through the local
UNANIMOUS-vote accept rule (resistance via beta) and measures the steady-state fraction
at Phi* (nu^beta). Concentration nu^beta -> 1 as beta -> 0 IFF the pool contains optimal
candidates -- so this also reports the pool's optimal fraction (the de-risk)."""
import sys, os, random as pyrandom
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from scaled_vec import init_P, train, rollout, _strides
from scale_spike2 import g_gen, u_gen
from scale_spike import closed_form_optimum
T, GAMMA = 16, 0.99

def hardened_stat(P, n, m, H, alpha, strides_t):
    _, rews, pots = rollout(P, tuple([0]*n), g_gen, u_gen, m, H, T, 1, GAMMA, alpha, strides_t, greedy=True)
    w = GAMMA ** torch.arange(T)
    rews_t = rews.squeeze(-1); pots_t = pots.squeeze(-1)           # (T,n),(T,)
    g_t = (rews_t.sum(1) - pots_t) / (n - 1)                       # recover g_t
    u_t = rews_t - g_t.unsqueeze(1)                                # (T,n)
    G = float((g_t * w).sum()); U = (u_t * w.unsqueeze(1)).sum(0).numpy()
    return (G, U, G + float(U.sum()))

def build_pool(k, ncand, episodes=2000, batch=64, H=1, alpha=0.1, lr=1e-3):
    m = k; strides_t = torch.tensor(_strides([m + 1] * H + [m])); init = tuple([0] * k)
    pool = []
    for c in range(ncand):
        pyrandom.seed(1000 + c); np.random.seed(1000 + c); torch.manual_seed(1000 + c)
        P = init_P(k, m, H)
        train(P, init, g_gen, u_gen, m, H, T, episodes, batch, GAMMA, lr, alpha, strides_t)
        pool.append(hardened_stat(P, k, m, H, alpha, strides_t))
    return pool

def accepts_n(sb, sa, beta, n, rng):
    Gb, Ub, _ = sb; Ga, Ua, _ = sa
    for i in range(n):
        Delta = (Gb - Ga) / n + (Ub[i] - Ua[i])
        if rng.random() >= min(1.0, np.exp(Delta / beta)):        # unanimous: all agents must accept
            return False
    return True

def hopping_chain(pool, beta, n, opt, epochs=4000, burn=400, seed=0):
    rng = pyrandom.Random(seed)
    inc = min(pool, key=lambda s: s[2])                           # start at the worst (deepest trap)
    at = 0
    for e in range(epochs):
        cand = pool[rng.randrange(len(pool))]
        if accepts_n(cand, inc, beta, n, rng): inc = cand
        if e >= burn and inc[2] / opt >= 0.99: at += 1
    return at / (epochs - burn)

if __name__ == "__main__":
    for k in [4]:
        opt, _ = closed_form_optimum(k)
        pool = build_pool(k, 60)
        ratios = np.array([s[2] / opt for s in pool])
        print(f"\nk={k}: pool={len(pool)}  optimal-frac(>=0.99)={np.mean(ratios>=0.99):.2f}  "
              f"mean={ratios.mean():.3f}  best={ratios.max():.3f}  worst={ratios.min():.3f}")
        if (ratios >= 0.99).any():
            print("  hopping concentration nu^beta(opt) vs temperature:")
            for beta in [0.3, 0.1, 0.05, 0.02, 0.01]:
                nu = np.mean([hopping_chain(pool, beta, k, opt, seed=s) for s in range(5)])
                print(f"    beta={beta:<5}: nu={nu:.3f}")
        else:
            print("  pool has NO optimal candidate -> hopping cannot concentrate on Phi* here.")
