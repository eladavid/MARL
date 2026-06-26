"""Does the GRADED-coverage game keep the optimum reachable as agents scale?
For each k: build a pool (graded rho, adequate episodes), normalize by the TRUE VI
optimum where feasible (else closed-form lower bound), report p_hat(reach Phi*) and the
ratio distribution. p_hat>0 with reasonable Q => hopping concentrates (reach ~1/p_hat).
Run: python3 graded_scaling.py --ks 5,6,7 --Q 150 --episodes 5000 --rho 0.0"""
import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import scaled_pool as sp, scale_spike
from graded_game import make_g_torch, closed_form, vi

def reference(k, rho, use_vi):
    # abstract k x k x k game: VI table is (k^k)^2 -> ~17GB at k=6 (the wall we document).
    # VI only feasible to k=5; use the closed-form Phi* (verified within 0.3% of VI) beyond.
    cf = closed_form(k, rho)
    if use_vi and k <= 5:
        t0 = time.time(); star, _ = vi(k, rho); dt = time.time() - t0
        return star, f"VI={star:.2f} (cf={cf:.2f}, {dt:.0f}s)"
    return cf, f"closed-form={cf:.2f} (VI infeasible: (k^k)^2 table)"

def run_k(k, rho, Q, episodes, batch, chunk, use_vi):
    ref, refstr = reference(k, rho, use_vi)
    sp.g_gen = make_g_torch(rho)
    scale_spike.closed_form_optimum = lambda n, T=16, gamma=0.99: (ref, None)
    sp.closed_form_optimum = scale_spike.closed_form_optimum
    t0 = time.time()
    r = sp.build_pool(k, Q, episodes=episodes, batch=batch, chunk=chunk, log=False)
    dt = time.time() - t0
    p99 = (r >= 0.99).mean(); p95 = (r >= 0.95).mean()
    print(f"k={k} rho={rho} Q={Q} ep={episodes} | ref {refstr}")
    print(f"   p(>=0.99)={p99:.3f}  p(>=0.95)={p95:.3f}  best={r.max():.3f} mean={r.mean():.3f}"
          f"  | reach~{1/p99:.0f} epochs if p99>0  ({dt:.0f}s)", flush=True)
    return p99

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", type=str, default="5,6,7")
    ap.add_argument("--Q", type=int, default=150)
    ap.add_argument("--episodes", type=int, default=5000)
    ap.add_argument("--rho", type=float, default=0.0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=150)
    ap.add_argument("--no-vi", action="store_true")
    a = ap.parse_args()
    print(f"GRADED scaling: basin reachability vs #agents (rho={a.rho})\n")
    for k in [int(x) for x in a.ks.split(",")]:
        run_k(k, a.rho, a.Q, a.episodes, a.batch, a.chunk, not a.no_vi)
