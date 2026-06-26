"""Pool-vectorized MAC-REINFORCE: train Q independent k-agent systems AT ONCE (extra
leading dim), to build large candidate pools fast. Question (per user): is p>0 at k=5,
i.e. does local search land in the optimal basin with ANY probability? ONE optimum in a
big pool proves p>0 -> hopping concentrates on it (reach time ~1/p). Faithful to
scaled_vec (direct tabular, fixed pad=m, reward-to-go + mean baseline)."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
torch.set_num_threads(1)
from congestion_game.policies import project_onto_simplex
from vectorized_train import disc_returns
from scaled_vec import _strides
from scale_spike2 import g_gen, u_gen
from scale_spike import closed_form_optimum
T, GAMMA = 16, 1.0

def _probs(P, aug, strides_t, m, alpha):
    Q, n, B = aug.shape[:3]
    idx = (aug * strides_t).sum(-1)                                  # (Q,n,B)
    p = P[torch.arange(Q)[:, None, None], torch.arange(n)[None, :, None], idx]   # (Q,n,B,m)
    p = torch.clamp(p, min=0.0); p = p / p.sum(-1, keepdim=True)
    return (1 - alpha) * p + alpha / m

def rollout(P, Q, n, m, H, B, alpha, strides_t, greedy=False):
    PAD = m
    curr = torch.zeros(Q, n, B, dtype=torch.long)                   # collided start (slot 0)
    buf = torch.full((Q, n, B, H), PAD, dtype=torch.long)
    logps, rews, pots = [], [], []
    for t in range(T):
        aug = torch.cat([buf, curr.unsqueeze(3)], dim=3)            # (Q,n,B,H+1)
        probs = _probs(P, aug, strides_t, m, alpha)
        if greedy:
            a = probs.argmax(-1); lp = torch.zeros(Q, n, B)
        else:
            d = torch.distributions.Categorical(probs); a = d.sample(); lp = d.log_prob(a)
        a_qbn = a.permute(0, 2, 1).reshape(Q * B, n)                # (Q*B,n)
        g = g_gen(a_qbn, n, m).reshape(Q, B)                        # (Q,B)
        u = u_gen(curr, a, m)                                       # (Q,n,B)
        rews.append(g.unsqueeze(1) + u); pots.append(g + u.sum(1)); logps.append(lp)
        if H > 0:
            buf = torch.cat([buf[:, :, :, 1:], curr.unsqueeze(3)], dim=3)
        curr = a
    return torch.stack(logps), torch.stack(rews), torch.stack(pots)  # (T,Q,n,B),(T,Q,n,B),(T,Q,B)

def build_pool(k, Q, episodes=2000, batch=32, H=1, alpha=0.1, lr=1e-3, seed=0, chunk=512, log=True):
    m = k; strides_t = torch.tensor(_strides([m + 1] * H + [m])); opt, _ = closed_form_optimum(k)
    ratios = []; t0 = time.time(); nchunk = (Q + chunk - 1) // chunk
    for ci, c0 in enumerate(range(0, Q, chunk)):
        q = min(chunk, Q - c0)
        torch.manual_seed(seed + c0)
        S = (m + 1) ** H * m
        P = torch.nn.Parameter(torch.distributions.Dirichlet(torch.ones(m)).sample((q, k, S)))
        optim = torch.optim.SGD([P], lr=lr)
        step = max(1, episodes // 10)
        for ep in range(episodes):
            logps, rews, _ = rollout(P, q, k, m, H, batch, alpha, strides_t)
            rets = disc_returns(rews.detach(), GAMMA)
            adv = rets - rets.mean(dim=3, keepdim=True)
            loss = -(logps * adv).sum() / batch
            optim.zero_grad(); loss.backward(); optim.step()
            with torch.no_grad():
                P.copy_(project_onto_simplex(P.view(q * k * S, m)).view(q, k, S, m))
            if log and (ep + 1) % step == 0:
                print(f"  [chunk {ci+1}/{nchunk} q={q}] ep {ep+1}/{episodes}  "
                      f"({time.time()-t0:.0f}s elapsed)", flush=True)
        with torch.no_grad():
            _, _, pots = rollout(P, q, k, m, H, 1, alpha, strides_t, greedy=True)   # (T,q,1)
            phi = (disc_returns(pots, GAMMA)[0]).squeeze(-1).numpy()                # (q,)
        ratios.extend((phi / opt).tolist())
        if log:
            r = np.array(ratios)
            print(f"  >> chunk {ci+1}/{nchunk} done | candidates so far={len(r)}  "
                  f"OPTIMA so far={int((r>=0.99).sum())}  best={r.max():.3f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
    return np.array(ratios)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5); ap.add_argument("--Q", type=int, default=500)
    ap.add_argument("--episodes", type=int, default=2000); ap.add_argument("--chunk", type=int, default=512)
    a = ap.parse_args()
    t0 = time.time()
    r = build_pool(a.k, a.Q, episodes=a.episodes, chunk=a.chunk)
    nopt = int((r >= 0.99).sum())
    print(f"k={a.k}  pool Q={a.Q}  episodes={a.episodes}  ({time.time()-t0:.0f}s)")
    print(f"  optimal (>=0.99): {nopt}/{a.Q}   p_hat={nopt/a.Q:.4f}   "
          f"best={r.max():.3f} mean={r.mean():.3f}")
    print(f"  -> {'p>0 CONFIRMED: hopping concentrates (reach ~1/p)' if nopt>0 else 'no optimum yet; go bigger Q'}")
