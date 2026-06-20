"""Scaling spike (Phase 2): MAC-REINFORCE on the generalized rotation-coverage game,
measured against the CLOSED-FORM optimum (no brute force needed).

Reuses the validated vectorized trainer by monkeypatching the generalized g/u
(exactly the pattern in u_compare.py). For each n: train n independent agents from
the collided start, harden, and report the achieved discounted potential as a ratio
of the closed-form Phi* (= VI optimum, cross-checked in scale_spike.py).
"""
import sys, os, time, random as pyrandom
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import vectorized_train as vt
from vectorized_train import train_vectorized
from congestion_game.policies import DirectTabularPolicy
from scale_spike import closed_form_optimum, BONUS, C, GAMMA, T

# ---- generalized game (vectorized), m = n slots --------------------------------
def g_gen(actions, N, A):
    counts = torch.nn.functional.one_hot(actions, A).sum(1).float()   # (B,A)
    is_perm = (counts <= 1).all(1)                                    # permutation <=> full coverage
    cong = 1.0 - ((counts / N) ** 2).sum(1)
    return torch.where(is_perm, torch.full_like(cong, BONUS), cong)

def u_gen(s, a, S):
    return C * (a != s).float()                                       # move-rewarding

def greedy_phi(pols, init, m, H):
    """Discounted Phi of the hardened (argmax) joint policy from the collided start."""
    N = len(pols)
    curr = torch.stack([torch.full((1,), int(x), dtype=torch.long) for x in init])  # (N,1)
    buf = torch.zeros(N, 1, H, dtype=torch.long); tot = 0.0
    with torch.no_grad():
        for t in range(T):
            acts = []
            for i, p in enumerate(pols):
                aug = torch.cat([buf[i], curr[i].unsqueeze(1)], dim=1)
                probs = p(aug).view(-1, m)
                acts.append(torch.argmax(probs, dim=1))
            actions = torch.stack(acts, dim=1)                        # (1,N)
            g = g_gen(actions, N, m)
            us = sum(u_gen(curr[i], actions[:, i], m) for i in range(N))
            tot += (GAMMA ** t) * float((g + us)[0])
            if H > 0:
                buf = torch.cat([buf[:, :, 1:], curr.unsqueeze(2)], dim=2)
            curr = actions.t().contiguous()
    return tot

def run_n(n, H=2, seeds=4, episodes=800, batch=64, lr=1e-3):
    vt.g_batched, vt.u_batched = g_gen, u_gen          # monkeypatch the generalized game
    m = n; init = tuple([0] * n)
    phistar, _ = closed_form_optimum(n)
    ratios, t0 = [], time.time()
    for seed in range(seeds):
        pyrandom.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        pols = [DirectTabularPolicy([m] * (H + 1), m, 0.1) for _ in range(n)]
        train_vectorized(pols, init, episodes, batch, m, m, H, T, GAMMA, lr, True)
        ratios.append(greedy_phi(pols, init, m, H) / phistar)
    r = np.array(ratios); dt = time.time() - t0
    print(f"n={n}  H={H}  Phi*={phistar:7.2f} | reach(>=0.99)={int((r>=0.99).sum())}/{seeds}  "
          f"best={r.max():.3f}  mean={r.mean():.3f}  | {dt:.1f}s ({dt/seeds:.1f}s/seed)", flush=True)
    return r

if __name__ == "__main__":
    from scale_spike import best_static_value
    print("k x k x k coverage-bonus game (scaled 3x3x3): reach rate vs k\n")
    for n in [3, 4, 5, 6]:
        stat = best_static_value(n, n); cf, _ = closed_form_optimum(n)
        print(f"  [k={n} static-trap ratio={stat/cf:.3f}]")
        run_n(n, H=2, seeds=6, episodes=2000, batch=64)
        print()
