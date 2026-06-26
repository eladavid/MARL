"""Does graded coverage widen the k=5 basin? Sweep sharpness rho, measure p_hat
(fraction of pool reaching Phi*) via the candidate-vectorized trainer. rho=1 was the
needle (0/1000). Looking for rho where p>~0.05 while the gap/traps persist."""
import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import scaled_pool as sp
from graded_game import make_g_torch, closed_form, C, W

def run(k, rho, Q, episodes, batch, chunk):
    # monkeypatch the pool trainer's g with the graded version
    sp.g_gen = make_g_torch(rho)
    opt = closed_form(k, rho)
    # closed_form expects n=k (m=n game); patch scale_spike.closed_form_optimum used inside build_pool
    import scale_spike
    _orig = scale_spike.closed_form_optimum
    scale_spike.closed_form_optimum = lambda n, T=16, gamma=0.99: (closed_form(n, rho), None)
    sp.closed_form_optimum = scale_spike.closed_form_optimum
    t0 = time.time()
    r = sp.build_pool(k, Q, episodes=episodes, batch=batch, chunk=chunk, log=False)
    scale_spike.closed_form_optimum = _orig
    nopt = int((r >= 0.99).sum())
    print(f"k={k} rho={rho:.2f} Q={Q}: optima={nopt}/{Q}  p_hat={nopt/Q:.4f}  "
          f"best={r.max():.3f} mean={r.mean():.3f}  ({time.time()-t0:.0f}s)", flush=True)
    return nopt / Q

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5); ap.add_argument("--Q", type=int, default=300)
    ap.add_argument("--episodes", type=int, default=2000); ap.add_argument("--chunk", type=int, default=300)
    a = ap.parse_args()
    print(f"basin width vs sharpness, k={a.k}, Q={a.Q}, ep={a.episodes} (rho=1 was needle 0/1000)\n")
    for rho in [0.0, 0.2, 0.4, 0.6]:
        run(a.k, rho, a.Q, a.episodes, 32, a.chunk)
