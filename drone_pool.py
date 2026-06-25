"""Candidate-vectorized MAC-REINFORCE for the drone game (train Q systems at once),
to build a large H=1 pool for the meta-algorithm. Pulls the n=4 config from drone_game.
Returns per-candidate hardened (Gbar, Ubar_vec, Phi) normalized like meta_algorithm, plus
the ratio to Phi*, so the validated accepts() can replay it (same recipe as meta_replay)."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
torch.set_num_threads(1)
from congestion_game.policies import project_onto_simplex
from vectorized_train import disc_returns
import drone_game as dg
from drone_train import g_batched, batt_step  # reuse the (config-live) reward/dynamics

def _cfg():
    return dg.R, dg.M, dg.NODES, dg.BATT, dg.B, dg.W_COV, dg.C_BATT, dg.GAMMA, dg.T

def rollout(P, Q, n, H, Bsz, alpha, strides_t, greedy=False):
    R, M, NODES, BATT, B, W_COV, C_BATT, GAMMA, T = _cfg()
    PAD = NODES * BATT
    node = torch.zeros(Q, n, Bsz, dtype=torch.long)            # start at reload R0
    batt = torch.full((Q, n, Bsz), B, dtype=torch.long)
    buf = torch.full((Q, n, Bsz, H), PAD, dtype=torch.long)
    logps, rews, pots = [], [], []
    for t in range(T):
        flat = node * BATT + batt                              # (Q,n,Bsz)
        aug = torch.cat([buf, flat.unsqueeze(3)], dim=3)       # (Q,n,Bsz,H+1)
        idx = (aug * strides_t).sum(-1)                        # (Q,n,Bsz)
        probs = P[torch.arange(Q)[:, None, None], torch.arange(n)[None, :, None], idx]  # (Q,n,Bsz,NODES)
        probs = torch.clamp(probs, min=0); probs = probs / probs.sum(-1, keepdim=True)
        probs = (1 - alpha) * probs + alpha / NODES
        if greedy:
            a = probs.argmax(-1); lp = torch.zeros(Q, n, Bsz)
        else:
            d = torch.distributions.Categorical(probs); a = d.sample(); lp = d.log_prob(a)
        empty = batt == 0                                       # frozen reload at b=0
        a = torch.where(empty, torch.zeros_like(a), a); lp = torch.where(empty, torch.zeros_like(lp), lp)
        a_qbn = a.permute(0, 2, 1).reshape(Q * Bsz, n)
        g = g_batched(a_qbn, n).reshape(Q, Bsz)                # (Q,Bsz)
        bp = batt_step(batt, a)
        u = -C_BATT * (1 - bp.float() / B)                     # (Q,n,Bsz)
        rews.append(g.unsqueeze(1) + u); pots.append(g + u.sum(1)); logps.append(lp)
        if H > 0:
            buf = torch.cat([buf[:, :, :, 1:], flat.unsqueeze(3)], dim=3)
        node, batt = a, bp
    return torch.stack(logps), torch.stack(rews), torch.stack(pots)

def strides_for(H):
    L = dg.NODES * dg.BATT; V = [L + 1] * H + [L]; s = [1] * len(V)
    for k in range(len(V) - 2, -1, -1): s[k] = s[k + 1] * V[k + 1]
    return torch.tensor(s), int(((L + 1) ** H) * L)

def build_pool(n, Q, star, H=1, episodes=4000, batch=32, lr=1e-3, alpha=0.1, seed=0, chunk=200, R_MAX=None):
    R, M, NODES, BATT, B, W_COV, C_BATT, GAMMA, T = _cfg()
    if R_MAX is None: R_MAX = W_COV + C_BATT * n            # rough scale so Gbar,Ubar in O(1)
    strides_t, S = strides_for(H)
    stats = []; ratios = []; t0 = time.time()
    for c0 in range(0, Q, chunk):
        q = min(chunk, Q - c0); torch.manual_seed(seed + c0)
        P = torch.nn.Parameter(torch.distributions.Dirichlet(torch.ones(NODES)).sample((q, n, S)))
        opt = torch.optim.SGD([P], lr=lr)
        for _ in range(episodes):
            logps, rews, _ = rollout(P, q, n, H, batch, alpha, strides_t)
            rets = disc_returns(rews.detach(), GAMMA); adv = rets - rets.mean(3, keepdim=True)
            loss = -(logps * adv).sum() / batch
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad(): P.copy_(project_onto_simplex(P.view(q * n * S, NODES)).view(q, n, S, NODES))
        with torch.no_grad():
            _, rews, pots = rollout(P, q, n, H, 1, alpha, strides_t, greedy=True)
            w = (GAMMA ** torch.arange(T)).view(T, 1)
            rt = rews.squeeze(-1); pt = pots.squeeze(-1)       # (T,q,n),(T,q)
            g_t = (rt.sum(2) - pt) / (n - 1); u_t = rt - g_t.unsqueeze(2)
            G = (g_t * w).sum(0).numpy(); U = (u_t * w.unsqueeze(2)).sum(0).numpy()
            Phi = (disc_returns(pots, GAMMA)[0]).squeeze(-1).numpy()
        nrm = T * R_MAX
        for j in range(q):
            stats.append((float(G[j] / nrm), U[j] / nrm, float(Phi[j]))); ratios.append(float(Phi[j] / star))
        print(f"  pool {len(stats)}/{Q}  optfrac={np.mean(np.array(ratios)>=0.99):.3f} "
              f"mean={np.mean(ratios):.3f}  ({time.time()-t0:.0f}s)", flush=True)
    return stats, np.array(ratios), R_MAX

if __name__ == "__main__":
    dg.R = {0}; dg.M = {1, 2, 3}; dg.NODES = 4; dg.BATT = dg.B + 1
    import drone_train as dt; dt.refresh()
    from drone_vi_vec import vi_vec
    n = 4; star, _ = vi_vec(n, tuple([(0, dg.B)] * n))
    print(f"n=4 drone Phi*={star:.2f}; building H=1 pool Q=50 (smoke test)", flush=True)
    stats, ratios, rmax = build_pool(n, 50, star, H=1, episodes=4000, chunk=50)
    print(f"smoke: optfrac={np.mean(ratios>=0.99):.3f} mean={ratios.mean():.3f} best={ratios.max():.3f}")
