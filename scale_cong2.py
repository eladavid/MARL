"""Phase 2 on the evasive-coverage congestion game: does MAC-REINFORCE recover the
rotating patrol? Measured against the closed-form Phi* (no brute force needed)."""
import sys, os, time, random as pyrandom
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import torch.nn.functional as F
import vectorized_train as vt
from vectorized_train import train_vectorized
from congestion_game.policies import DirectTabularPolicy
from scale_cong import closed_form_optimum, best_static_value, joint_vi, W, C, GAMMA, T

def g_gen(actions, N, A):
    counts = F.one_hot(actions, A).sum(1).float()
    return W * (1.0 - ((counts / N) ** 2).sum(1))

def u_gen(s, a, S):
    return C * (a != s).float()

def greedy_phi(pols, init, m, H):
    N = len(pols)
    curr = torch.stack([torch.full((1,), int(x), dtype=torch.long) for x in init])
    buf = torch.zeros(N, 1, H, dtype=torch.long); tot = 0.0
    with torch.no_grad():
        for t in range(T):
            acts = []
            for i, p in enumerate(pols):
                aug = torch.cat([buf[i], curr[i].unsqueeze(1)], dim=1)
                acts.append(torch.argmax(p(aug).view(-1, m), dim=1))
            actions = torch.stack(acts, dim=1)
            g = g_gen(actions, N, m)
            us = sum(u_gen(curr[i], actions[:, i], m) for i in range(N))
            tot += (GAMMA ** t) * float((g + us)[0])
            if H > 0:
                buf = torch.cat([buf[:, :, 1:], curr.unsqueeze(2)], dim=2)
            curr = actions.t().contiguous()
    return tot

def run_n(n, m=3, H=2, seeds=6, episodes=2000, batch=64, lr=1e-3):
    vt.g_batched, vt.u_batched = g_gen, u_gen
    init = tuple([0] * n)
    phistar, _ = closed_form_optimum(n, m)
    ratios, t0 = [], time.time()
    for seed in range(seeds):
        pyrandom.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        pols = [DirectTabularPolicy([m] * (H + 1), m, 0.1) for _ in range(n)]
        train_vectorized(pols, init, episodes, batch, m, m, H, T, GAMMA, lr, True)
        ratios.append(greedy_phi(pols, init, m, H) / phistar)
    r = np.array(ratios); dt = time.time() - t0
    print(f"n={n} m={m} H={H} ep={episodes} | Phi*={phistar:6.1f} | "
          f"reach(>=0.99)={int((r>=0.99).sum())}/{seeds} best={r.max():.3f} "
          f"mean={r.mean():.3f} | {dt/seeds:.1f}s/seed", flush=True)
    return r

if __name__ == "__main__":
    print("does the SELECTION problem intensify with n? (reach rate vs n)\n")
    for n in [5, 6, 7, 8]:
        run_n(n, m=3, H=2, seeds=6, episodes=2000, batch=64)
