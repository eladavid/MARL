"""Hopping concentration via POOL REPLAY through the VALIDATED accept rule.

Insight (user): meta_algorithm.run_hopping random-RESTARTS each epoch (randomize -> train
-> harden -> candidate), so a hopping chain = {draw candidate from the random-restart
distribution; apply accept test; update incumbent}. Our fast vectorized pool IS that
distribution. So replay the pool through meta_algorithm.accepts (THE validated function),
normalizing each candidate's (G, U_i) exactly as meta_algorithm.eval_hardened does
(/(T*R_MAX)). No retraining (the slow episodic part), 100% validated decision logic.

This fixes the earlier bug: my hand-rolled accept omitted the R_MAX normalization, so
exp(Delta/beta) saturated and the curve was non-monotone. Using eval_hardened's scaling
+ the real accepts restores it."""
import sys, os, pickle, time, argparse, random as pyrandom
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
torch.set_num_threads(1)
import meta_algorithm as M                              # for THE validated accepts()
from congestion_game.policies import project_onto_simplex
from vectorized_train import disc_returns
from scaled_pool import rollout as prollout, _strides
import scaled_pool
from graded_game import make_g_torch, closed_form, C as MOVE_C, W as COV_W
T, GAMMA = 16, 1.0
R_MAX = COV_W + MOVE_C                                   # so Gbar,Ubar in (0,1), as in eval_hardened

def build_pool_stats(k, rho, Q, episodes=5000, batch=32, H=1, alpha=0.1, lr=1e-3, seed=0):
    """Train Q candidates at once (fast). Return NORMALIZED (Gbar, Ubar_vec, Phi) per
    candidate, matching meta_algorithm.eval_hardened's discounted + /(T*R_MAX) scaling."""
    scaled_pool.g_gen = make_g_torch(rho)
    m = k; strides_t = torch.tensor(_strides([m + 1] * H + [m])); S = (m + 1) ** H * m
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
        _, rews, pots = prollout(P, Q, k, m, H, 1, alpha, strides_t, greedy=True)
        w = (GAMMA ** torch.arange(T)).view(T, 1)
        rt = rews.squeeze(-1); pt = pots.squeeze(-1)
        g_t = (rt.sum(2) - pt) / (k - 1)                # (T,Q)
        u_t = rt - g_t.unsqueeze(2)                     # (T,Q,n)
        G = (g_t * w).sum(0).numpy()                    # (Q,)  discounted
        U = (u_t * w.unsqueeze(2)).sum(0).numpy()       # (Q,n) discounted
        Phi = (disc_returns(pots, GAMMA)[0]).squeeze(-1).numpy()   # (Q,) discounted Phi
    norm = T * R_MAX
    return [(float(G[q] / norm), (U[q] / norm), float(Phi[q])) for q in range(Q)]

def chain(pool, beta, n, opt, epochs=4000, burn=400, seed=0):
    """Replay: draw uniform candidate (= random restart), validated accept test, update."""
    M.N = n                                             # accepts loops range(M.N)
    rng_state = pyrandom.getstate(); pyrandom.seed(seed)
    inc = min(pool, key=lambda s: s[2])                 # start at worst (deepest trap)
    at = 0
    for e in range(epochs):
        cand = pool[pyrandom.randrange(len(pool))]
        if M.accepts(cand, inc, beta, reduced=True):    # THE validated accept rule
            inc = cand
        if e >= burn and inc[2] / opt >= 0.99:
            at += 1
    pyrandom.setstate(rng_state)
    return at / (epochs - burn)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5); ap.add_argument("--rho", type=float, default=0.7)
    ap.add_argument("--Q", type=int, default=200); ap.add_argument("--episodes", type=int, default=5000)
    ap.add_argument("--betas", type=str, default="0.3,0.15,0.08,0.04,0.02,0.01")
    a = ap.parse_args()
    opt = closed_form(a.k, a.rho); t0 = time.time()
    pool = build_pool_stats(a.k, a.rho, a.Q, episodes=a.episodes)
    ratios = np.array([s[2] / opt for s in pool])
    print(f"pool k={a.k} rho={a.rho} Q={a.Q}: optfrac={np.mean(ratios>=0.99):.3f} "
          f"mean={ratios.mean():.3f} ({time.time()-t0:.0f}s build)", flush=True)
    betas = [float(b) for b in a.betas.split(",")]
    nu = {b: float(np.mean([chain(pool, b, a.k, opt, seed=s) for s in range(8)])) for b in betas}
    for b in betas:
        print(f"  beta={b:<5}: nu^beta(opt) = {nu[b]:.3f}", flush=True)
    pickle.dump({"k": a.k, "rho": a.rho, "betas": betas, "nu": nu,
                 "pool_optfrac": float(np.mean(ratios >= 0.99)),
                 "pool_ratios": ratios.tolist()}, open("sim_figs/hopping_data.pkl", "wb"))
    print("saved sim_figs/hopping_data.pkl")
