"""Remote pool worker (thread-pinned, chunked, resumable).

Trains a slice of MAC-REINFORCE candidates on the n=4 DRONE game and saves their
hardened (Gbar, Ubar_vec, Phi) stats + ratio-to-Phi* incrementally per chunk.

Reuses the TESTED code paths (drone_pool.build_pool logic, drone_game config).
Thread pinning (the critical fix) is set BELOW before numpy/torch import AND should
also be exported in the shell (run_all.sh does both).

Usage:
  python3 remote/pool_worker.py --offset 0 --count 2000 --episodes 6000 \
      --chunk 250 --H 1 --batch sched --out results_remote
Args:
  --batch  fixed   -> fixed batch size 64
           sched   -> schedule 4->8->16->32->64->128 (variance control)
  --H      history buffer length (1 = sufficient to represent the optimum; 0 = memoryless ceiling)
"""
import os
for _v in ["OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"]:
    os.environ[_v] = "1"
import sys, time, pickle, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # MARL repo root
sys.path.insert(0, ROOT)
import numpy as np, torch
torch.set_num_threads(1)
import drone_game as dg
# --- canonical n=4 drone config: 1 charger (R0) + 3 missions (M1,M2,M3), 4 nodes ---
dg.R = {0}; dg.M = {1, 2, 3}; dg.NODES = 4; dg.BATT = dg.B + 1
from congestion_game.policies import project_onto_simplex
from vectorized_train import disc_returns
import drone_train as dt; dt.refresh()
import drone_pool as dp
from drone_vi_vec import vi_vec

SCHED = [4, 8, 16, 32, 64, 128]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offset", type=int, required=True)
    ap.add_argument("--count", type=int, required=True)
    ap.add_argument("--episodes", type=int, default=6000)
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--H", type=int, default=1)
    ap.add_argument("--batch", choices=["fixed", "sched"], default="sched")
    ap.add_argument("--fixed-batch", type=int, default=64, help="batch size when --batch fixed")
    ap.add_argument("--out", type=str, default="results_remote")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    n = 4; H = a.H
    star, _ = vi_vec(n, tuple([(0, dg.B)] * n))
    strides_t, S = dp.strides_for(H); NODES = dg.NODES
    rmax = dg.W_COV + dg.C_BATT * n
    def batch_at(ep):
        return SCHED[min(len(SCHED) - 1, ep * len(SCHED) // a.episodes)] if a.batch == "sched" else a.fixed_batch

    # --- resume: load this worker's checkpoint and skip already-finished chunks ---
    tag = a.batch if a.batch == "sched" else f"fixed{a.fixed_batch}"
    outpath = os.path.join(a.out, f"part_H{H}_{tag}_{a.offset}.pkl")
    allstats, allratios = [], []
    if os.path.exists(outpath):
        try:
            prev = pickle.load(open(outpath, "rb"))
            allstats = list(prev["stats"]); allratios = list(prev["ratios"])
            print(f"[worker {a.offset} H{H} {tag}] resume: {len(allratios)}/{a.count} already done", flush=True)
        except Exception as e:
            allstats, allratios = [], []
            print(f"[worker {a.offset} H{H} {tag}] checkpoint unreadable ({e}); starting fresh", flush=True)
    done = len(allratios)
    t0 = time.time()
    for c0 in range(0, a.count, a.chunk):
        if c0 < done:                       # chunk already checkpointed (seeds are deterministic per c0) -> skip
            continue
        q = min(a.chunk, a.count - c0)
        torch.manual_seed(a.offset + c0)
        P = torch.nn.Parameter(torch.distributions.Dirichlet(torch.ones(NODES)).sample((q, n, S)))
        opt = torch.optim.SGD([P], lr=1e-3)
        for ep in range(a.episodes):
            b = batch_at(ep)
            logps, rews, _ = dp.rollout(P, q, n, H, b, 0.1, strides_t)
            rets = disc_returns(rews.detach(), dg.GAMMA); adv = rets - rets.mean(3, keepdim=True)
            loss = -(logps * adv).sum() / b
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                P.copy_(project_onto_simplex(P.view(q * n * S, NODES)).view(q, n, S, NODES))
        with torch.no_grad():
            _, rews, pots = dp.rollout(P, q, n, H, 1, 0.1, strides_t, greedy=True)
            w = (dg.GAMMA ** torch.arange(dg.T)).view(dg.T, 1); rt = rews.squeeze(-1); pt = pots.squeeze(-1)
            g_t = (rt.sum(2) - pt) / (n - 1); u_t = rt - g_t.unsqueeze(2)
            G = (g_t * w).sum(0).numpy(); U = (u_t * w.unsqueeze(2)).sum(0).numpy()
            Phi = (disc_returns(pots, dg.GAMMA)[0]).squeeze(-1).numpy()
        nrm = dg.T * rmax
        for j in range(q):
            allstats.append((float(G[j] / nrm), U[j] / nrm, float(Phi[j]))); allratios.append(float(Phi[j] / star))
        arr = np.array(allratios)
        with open(outpath + ".tmp", "wb") as fh:
            pickle.dump({"stats": allstats, "ratios": arr.tolist(), "star": star, "R_MAX": rmax,
                         "n": n, "H": H, "batch": a.batch, "fixed_batch": a.fixed_batch, "offset": a.offset}, fh)
        os.replace(outpath + ".tmp", outpath)   # atomic: never leaves a half-written checkpoint
        print(f"[worker {a.offset} H{H} {a.batch}] {len(allstats)}/{a.count}  "
              f"optima(>=0.99)={int((arr>=0.99).sum())}  best={arr.max():.4f}  mean={arr.mean():.3f}  "
              f"({time.time()-t0:.0f}s)", flush=True)

if __name__ == "__main__":
    main()
