"""How sharp is the learned policy vs training episodes? Tracks, at checkpoints:
  hard Phi/Phi* (argmax) , soft Phi/Phi* (alpha=0 stochastic) ,
  mean max-prob and entropy of pi on the states actually VISITED on the greedy path
  (excluding frozen empty-battery states, where the action is forced).
Answers: 'are the probabilities still low / still converging at 10K?'
"""
import os
for _v in ["OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"]:
    os.environ[_v] = "1"
import sys, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np, torch, random as pyrandom
torch.set_num_threads(1)
import drone_game as dg
dg.R = {0}; dg.M = {1, 2, 3}; dg.NODES = 4; dg.BATT = dg.B + 1
import drone_train as dt; dt.refresh()
from drone_vi_vec import vi_vec
from congestion_game.policies import project_onto_simplex
from vectorized_train import disc_returns

def sharpness(P, n, H, strides_t):
    NODES, BATT, B, T = dt.NODES, dt.BATT, dt.B, dg.T
    PAD = NODES * BATT
    node = torch.zeros(n, 1, dtype=torch.long); batt = torch.full((n, 1), B, dtype=torch.long)
    buf = torch.full((n, 1, H), PAD, dtype=torch.long)
    maxps, ents = [], []
    for t in range(T):
        flat = node * BATT + batt
        aug = torch.cat([buf, flat.unsqueeze(2)], dim=2); idx = (aug * strides_t).sum(-1)
        probs = P[torch.arange(n).unsqueeze(1), idx].clamp(min=0)
        probs = probs / probs.sum(-1, keepdim=True)                    # raw pi, alpha=0
        mp = probs.max(-1).values; ent = -(probs * probs.clamp(min=1e-12).log()).sum(-1)
        empty = (batt == 0)
        for ag in range(n):
            if not bool(empty[ag, 0]): maxps.append(float(mp[ag, 0])); ents.append(float(ent[ag, 0]))
        a = torch.where(empty, torch.zeros_like(probs.argmax(-1)), probs.argmax(-1))
        bp = dt.batt_step(batt, a)
        if H > 0: buf = torch.cat([buf[:, :, 1:], flat.unsqueeze(2)], dim=2)
        node, batt = a, bp
    return float(np.mean(maxps)), float(np.mean(ents))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--H", type=int, default=1)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--checkpoints", type=int, nargs="+", default=[500, 1000, 2000, 5000, 10000])
    a = ap.parse_args()
    n = 4; H = a.H; star, _ = vi_vec(n, tuple([(0, dg.B)] * n))
    pyrandom.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    strides_t, S = dt.strides_for(H)
    P = torch.nn.Parameter(torch.distributions.Dirichlet(torch.ones(dg.NODES)).sample((n, S)))
    opt = torch.optim.SGD([P], lr=a.lr)
    def hardphi():
        with torch.no_grad():
            _, _, pots = dt.rollout(P, n, H, 1, 0.0, strides_t, greedy=True)
            return float(disc_returns(pots.detach(), dg.GAMMA)[0].mean()) / star
    def softphi():
        with torch.no_grad():
            _, _, pots = dt.rollout(P, n, H, 512, 0.0, strides_t, greedy=False)
            return float(disc_returns(pots.detach(), dg.GAMMA)[0].mean()) / star
    print(f"Phi*={star:.2f}  seed={a.seed} batch={a.batch} (uniform max-prob baseline = {1/dg.NODES:.2f})")
    print(f"{'ep':>6} {'hardPhi':>8} {'softPhi':>8} {'maxprob':>8} {'entropy':>8}")
    cps = sorted(set(a.checkpoints)); ci = 0; total = cps[-1]
    for ep in range(1, total + 1):
        logps, rews, _ = dt.rollout(P, n, H, a.batch, a.alpha, strides_t)
        rets = disc_returns(rews.detach(), dg.GAMMA); adv = rets - rets.mean(2, keepdim=True)
        loss = -(logps * adv).sum() / a.batch
        opt.zero_grad(); loss.backward()
        if P.grad is not None and torch.isfinite(P.grad).all():
            torch.nn.utils.clip_grad_norm_([P], 10.0); opt.step()
        else: opt.zero_grad()
        with torch.no_grad():
            Pn = project_onto_simplex(P.view(n * S, dg.NODES)).view(n, S, dg.NODES)
            Pn = torch.nan_to_num(Pn, nan=1.0 / dg.NODES).clamp(min=0); P.copy_(Pn / Pn.sum(-1, keepdim=True))
        if ci < len(cps) and ep == cps[ci]:
            mp, en = sharpness(P, n, H, strides_t)
            print(f"{ep:>6} {hardphi():>8.3f} {softphi():>8.3f} {mp:>8.3f} {en:>8.3f}", flush=True)
            ci += 1

if __name__ == "__main__":
    main()
