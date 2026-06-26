"""Remote worker: build a graded-game candidate pool at agent-count k (thread-pinned,
chunked, resumable). The abstract congestion game has a CLOSED-FORM optimum, so we can
scale agents freely and still know Phi* exactly -- this is where the 'scaling in n' claim
is earned. Saves per-candidate normalized (Gbar, Ubar, Phi) stats for the validated
meta_algorithm.accepts replay (abstract_analyze.py).

Usage: python3 remote/abstract_worker.py --k 6 --offset 0 --count 2000 \
           --episodes 5000 --chunk 250 --rho 0.7 --out results_abstract
"""
import os
for _v in ["OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"]:
    os.environ[_v] = "1"
import sys, time, pickle, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np, torch
torch.set_num_threads(1)
from meta_replay import build_pool_stats          # graded-game pool (R_MAX-normalized for accepts)
from graded_game import closed_form

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--count", type=int, required=True)
    ap.add_argument("--episodes", type=int, default=5000)   # k>=5 needs ~5000 (2000 undertrains)
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--rho", type=float, default=0.7)
    ap.add_argument("--out", type=str, default="results_abstract")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    opt = closed_form(a.k, a.rho)
    allstats = []; t0 = time.time()
    for c0 in range(0, a.count, a.chunk):
        q = min(a.chunk, a.count - c0)
        st = build_pool_stats(a.k, a.rho, q, episodes=a.episodes, seed=a.offset + c0)
        allstats += st
        ratios = np.array([s[2] / opt for s in allstats])
        pickle.dump({"stats": allstats, "ratios": ratios.tolist(), "opt": opt,
                     "k": a.k, "rho": a.rho, "offset": a.offset},
                    open(os.path.join(a.out, f"abs_k{a.k}_{a.offset}.pkl"), "wb"))
        print(f"[k={a.k} w{a.offset}] {len(allstats)}/{a.count}  "
              f"optfrac(>=0.99)={np.mean(ratios>=0.99):.4f}  best={ratios.max():.4f}  "
              f"mean={ratios.mean():.3f}  ({time.time()-t0:.0f}s)", flush=True)

if __name__ == "__main__":
    main()
