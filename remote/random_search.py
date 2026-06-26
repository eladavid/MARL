"""Random-deterministic baseline ("early-hardening"): sample N random deterministic
policies (argmax of a random Dirichlet table), evaluate hardened Phi/Phi*, report the
best + how many reach the optimum. Training-free, so millions are cheap.

This makes the paper's "0 of 1e6 random policies reach the optimum" claim a saved,
reproducible artifact (random_search.json).

Usage: python3 remote/random_search.py --total 1000000 --batch 25000 --H 1 --out results_remote
"""
import os
for _v in ["OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"]:
    os.environ[_v] = "1"
import sys, json, time, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np, torch
torch.set_num_threads(1)
import drone_game as dg
dg.R = {0}; dg.M = {1, 2, 3}; dg.NODES = 4; dg.BATT = dg.B + 1
import drone_pool as dp
from drone_vi_vec import vi_vec
from vectorized_train import disc_returns

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=1_000_000)
    ap.add_argument("--batch", type=int, default=25000)
    ap.add_argument("--H", type=int, default=1)
    ap.add_argument("--out", type=str, default="results_remote")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    n = 4; H = a.H
    star, _ = vi_vec(n, tuple([(0, dg.B)] * n))
    strides_t, S = dp.strides_for(H); NODES = dg.NODES
    best = -1e9; nopt = 0; n95 = 0; total = 0; t0 = time.time()
    nb = (a.total + a.batch - 1) // a.batch
    for b in range(nb):
        torch.manual_seed(1234 + b)
        P = torch.distributions.Dirichlet(torch.ones(NODES)).sample((a.batch, n, S))  # random -> argmax = random det policy
        with torch.no_grad():
            _, _, pots = dp.rollout(P, a.batch, n, H, 1, 0.0, strides_t, greedy=True)
            r = (disc_returns(pots, dg.GAMMA)[0]).squeeze(-1).numpy() / star
        total += len(r); best = max(best, float(r.max()))
        nopt += int((r >= 0.99).sum()); n95 += int((r >= 0.95).sum())
        if (b + 1) % 5 == 0 or b == nb - 1:
            print(f"  {total} sampled: best={best:.4f} optima(>=0.99)={nopt} >=0.95={n95} ({time.time()-t0:.0f}s)", flush=True)
    res = {"total": total, "best": best, "optima_99": nopt, "ge_95": n95, "H": H, "star": star}
    json.dump(res, open(os.path.join(a.out, "random_search.json"), "w"), indent=2)
    print(f"\nRANDOM SEARCH: {total} policies, best={best:.4f}, optima={nopt}, >=0.95={n95}  -> {a.out}/random_search.json")

if __name__ == "__main__":
    main()
