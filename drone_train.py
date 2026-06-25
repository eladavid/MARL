"""Vectorized MAC-REINFORCE on the drone game (state = node x battery).
Per-agent flat state = node*BATT + batt in [0,25). H=0 = memoryless on (node,battery);
H>0 adds a pad-token history buffer. Independent per-agent policy gradient (direct tabular,
alpha-greedy, simplex projection). Measures hardened Phi vs the VI optimum."""
import sys, os, time, random as pyrandom
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
torch.set_num_threads(1)
from congestion_game.policies import project_onto_simplex
from vectorized_train import disc_returns
import drone_game as dg
from drone_game import B, GAMMA, T

# config pulled LIVE from drone_game (so n=4 4-node config is picked up); refresh() re-reads it
def refresh():
    global R, M, NODES, BATT, W_COV, C_BATT, W_CONG, RSET, MSET, NMISS
    R, M, NODES, BATT = dg.R, dg.M, dg.NODES, dg.BATT
    W_COV, C_BATT, W_CONG = dg.W_COV, dg.C_BATT, dg.W_CONG
    RSET = torch.tensor([1 if v in R else 0 for v in range(NODES)])
    MSET = torch.tensor([1 if v in M else 0 for v in range(NODES)])
    NMISS = len(M)
refresh()

def g_batched(actions, n):                              # actions (Bn, n) -> (Bn,)
    onehot = torch.nn.functional.one_hot(actions, NODES).sum(1).float()   # (Bn,NODES) counts
    miss_cov = ((onehot > 0).float() * MSET).sum(1)                       # distinct missions
    cong = torch.clamp(onehot - 1, min=0).sum(1)                          # stacking beyond 1/node
    return W_COV * miss_cov / NMISS - W_CONG * cong

def g_batched(actions, n):                              # actions (Bn, n) -> (Bn,)
    onehot = torch.nn.functional.one_hot(actions, NODES).sum(1).float()   # (Bn,NODES) counts
    miss_cov = ((onehot > 0).float() * MSET).sum(1)                       # distinct missions
    cong = torch.clamp(onehot - 1, min=0).sum(1)                          # stacking beyond 1/node
    return W_COV * miss_cov / NMISS - W_CONG * cong

def batt_step(batt, a):                                 # (.,) -> (.,)
    recharge = RSET[a].bool()
    return torch.where(recharge, torch.full_like(batt, B), torch.clamp(batt - 1, min=0))

def rollout(P, n, H, Bsz, alpha, strides_t, greedy=False):
    PAD = NODES * BATT                                  # pad token for history (= #local states)
    node = torch.zeros(n, Bsz, dtype=torch.long)        # start node 0 (reload R0)
    batt = torch.full((n, Bsz), B, dtype=torch.long)    # full battery
    buf = torch.full((n, Bsz, H), PAD, dtype=torch.long)
    logps, rews, pots = [], [], []
    for t in range(T):
        flat = node * BATT + batt                       # (n,Bsz) in [0,25)
        aug = torch.cat([buf, flat.unsqueeze(2)], dim=2)              # (n,Bsz,H+1)
        idx = (aug * strides_t).sum(-1)                              # (n,Bsz)
        probs = P[torch.arange(n).unsqueeze(1), idx]                 # (n,Bsz,NODES)
        probs = torch.clamp(probs, min=0); probs = probs / probs.sum(-1, keepdim=True)
        probs = (1 - alpha) * probs + alpha / NODES
        if greedy:
            a = probs.argmax(-1); lp = torch.zeros(n, Bsz)
        else:
            d = torch.distributions.Categorical(probs); a = d.sample(); lp = d.log_prob(a)
        # frozen (non-parameterized) sub-policy at empty battery: force reload (R0), no gradient
        empty = batt == 0
        a = torch.where(empty, torch.zeros_like(a), a)
        lp = torch.where(empty, torch.zeros_like(lp), lp)
        a_bn = a.t().contiguous()                                    # (Bsz,n)
        g = g_batched(a_bn, n)                                       # (Bsz,)
        bp = batt_step(batt, a)                                      # (n,Bsz) post-action battery
        u = -C_BATT * (1 - bp.float() / B)                          # (n,Bsz)
        rews.append(g.unsqueeze(0) + u); pots.append(g + u.sum(0)); logps.append(lp)
        if H > 0:
            buf = torch.cat([buf[:, :, 1:], flat.unsqueeze(2)], dim=2)
        node, batt = a, bp
    return torch.stack(logps), torch.stack(rews), torch.stack(pots)

def strides_for(H):
    L = NODES * BATT                                     # # local states; pad token = L (so L+1 slots)
    V = [L + 1] * H + [L]
    s = [1] * len(V)
    for k in range(len(V) - 2, -1, -1): s[k] = s[k + 1] * V[k + 1]
    return torch.tensor(s), int(((L + 1) ** H) * L)

def train_seed(n, H, episodes, batch, lr, alpha, seed):
    pyrandom.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    strides_t, S = strides_for(H)
    P = torch.nn.Parameter(torch.distributions.Dirichlet(torch.ones(NODES)).sample((n, S)))
    opt = torch.optim.SGD([P], lr=lr)
    for _ in range(episodes):
        logps, rews, _ = rollout(P, n, H, batch, alpha, strides_t)
        rets = disc_returns(rews.detach(), GAMMA); adv = rets - rets.mean(2, keepdim=True)  # baseline over BATCH (dim 2), not agents
        loss = -(logps * adv).sum() / batch
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            P.copy_(project_onto_simplex(P.view(n * S, NODES)).view(n, S, NODES))
    with torch.no_grad():
        _, _, pots = rollout(P, n, H, 1, alpha, strides_t, greedy=True)
        return float(disc_returns(pots.detach(), GAMMA)[0].mean())

if __name__ == "__main__":
    n = 3; s0 = tuple([(0, B)] * n)
    star, _ = vi(n, s0)
    print(f"drone MAC-REINFORCE n={n}: Phi*(VI)={star:.2f}\n")
    for H in [0, 1]:
        t0 = time.time(); ratios = []
        for seed in range(8):
            phi = train_seed(n, H, 3000, 64, 1e-3, 0.1, seed)
            ratios.append(phi / star)
        r = np.array(ratios)
        print(f"  H={H}: reach(>=0.99)={int((r>=0.99).sum())}/8  best={r.max():.3f} "
              f"mean={r.mean():.3f}  ({time.time()-t0:.0f}s)", flush=True)
