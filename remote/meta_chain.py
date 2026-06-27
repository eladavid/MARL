"""One resilient meta-algorithm (Nash-hopping) chain over the drone H=1 pool.
Replays random-restart candidates through the validated accept rule (meta_algorithm.accepts)
for one (beta, seed); writes nu = time-avg incumbent Phi/Phi* to a json (resumable: skips if done).
Short single process (~1-3s) so it completes before the flaky box faults; wrap in retry.
"""
import os
for _v in ["OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"]:
    os.environ[_v] = "1"
import sys, glob, json, pickle, argparse, random as pyrandom
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import drone_game as dg
dg.R = {0}; dg.M = {1, 2, 3}; dg.NODES = 4; dg.BATT = dg.B + 1
import meta_algorithm as M
M.N = 4  # match the n=4 drone game (analyze.py does the same)

def load_pool(dirp, H=1):
    stats = []; star = None
    for p in glob.glob(os.path.join(dirp, f"part_H{H}_*.pkl")):
        try:
            d = pickle.load(open(p, "rb")); stats += d["stats"]; star = d["star"]
        except Exception:
            pass
    return stats, star

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results_remote")
    ap.add_argument("--beta", type=float, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=300000)
    ap.add_argument("--burn", type=int, default=2000)
    ap.add_argument("--out", default="results_meta_drone")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fn = os.path.join(a.out, f"nu_b{a.beta}_s{a.seed}.json")
    if os.path.exists(fn):
        print(f"skip (done): {fn}"); return
    stats, star = load_pool(a.dir, 1)
    rng = pyrandom.Random(a.seed)
    inc = min(stats, key=lambda s: s[2])           # cold start at worst (matches meta_curve)
    vals = []; opt_draws = 0
    for e in range(a.epochs):
        c = stats[rng.randrange(len(stats))]        # random restart candidate (indep of incumbent)
        if c[2] / star >= 0.99: opt_draws += 1
        if M.accepts(c, inc, a.beta, True): inc = c
        if e >= a.burn: vals.append(inc[2] / star)
    res = {"beta": a.beta, "seed": a.seed, "epochs": a.epochs, "nu": float(np.mean(vals)),
           "opt_draws": opt_draws, "final_incumbent": inc[2] / star, "n_pool": len(stats)}
    tmp = fn + ".tmp"; json.dump(res, open(tmp, "w")); os.replace(tmp, fn)
    print(f"beta={a.beta} seed={a.seed}: nu={res['nu']:.4f} opt_draws={opt_draws} final={res['final_incumbent']:.4f}", flush=True)

if __name__ == "__main__":
    main()
