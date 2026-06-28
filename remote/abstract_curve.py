"""Learning-curve probe for the abstract graded game: train a small pool at agent-count k
and log the hardened-ratio DISTRIBUTION (mean / max / optfrac) every --eval-every episodes.
Answers (1) is the climb steady or staircase, and (2) the episodes confound -- run k=5 long
and watch its optfrac climb from ~6% (5K) toward the higher-k levels.
Mirrors meta_replay.build_pool_stats' training loop, adding periodic greedy eval.
"""
import os
for _v in ["OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"]:
    os.environ[_v] = "1"
import sys, json, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np, torch
torch.set_num_threads(1)
from congestion_game.policies import project_onto_simplex
from vectorized_train import disc_returns
from scaled_pool import rollout as prollout, _strides
import scaled_pool
from graded_game import make_g_torch, closed_form
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
T, GAMMA = 16, 1.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--Q", type=int, default=200)
    ap.add_argument("--episodes", type=int, required=True)
    ap.add_argument("--eval-every", type=int, default=2500)
    ap.add_argument("--rho", type=float, default=0.7)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", type=str, default="results_abstract")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    scaled_pool.g_gen = make_g_torch(a.rho)
    m = a.k; H = 1
    strides_t = torch.tensor(_strides([m + 1] * H + [m])); S = (m + 1) ** H * m
    opt = closed_form(a.k, a.rho)
    torch.manual_seed(0)
    P = torch.nn.Parameter(torch.distributions.Dirichlet(torch.ones(m)).sample((a.Q, a.k, S)))
    optim = torch.optim.SGD([P], lr=a.lr)

    def evalr():
        with torch.no_grad():
            _, _, pots = prollout(P, a.Q, a.k, m, H, 1, a.alpha, strides_t, greedy=True)
            Phi = (disc_returns(pots, GAMMA)[0]).squeeze(-1).numpy()
        return Phi / opt

    rec = []
    def snap(ep):
        r = evalr()
        rec.append({"ep": ep, "mean": float(r.mean()), "max": float(r.max()),
                    "optfrac": float((r >= 0.99).mean()), "ge95": float((r >= 0.95).mean())})
        print(f"k={a.k} ep={ep:>6}: mean={r.mean():.3f} max={r.max():.4f} optfrac={(r>=0.99).mean():.4f} >=.95={(r>=0.95).mean():.3f}", flush=True)

    snap(0)
    for ep in range(1, a.episodes + 1):
        logps, rews, _ = prollout(P, a.Q, a.k, m, H, a.batch, a.alpha, strides_t)
        rets = disc_returns(rews.detach(), GAMMA); adv = rets - rets.mean(dim=3, keepdim=True)
        loss = -(logps * adv).sum() / a.batch
        optim.zero_grad(); loss.backward(); optim.step()
        with torch.no_grad():
            P.copy_(project_onto_simplex(P.view(a.Q * a.k * S, m)).view(a.Q, a.k, S, m))
        if ep % a.eval_every == 0:
            snap(ep)

    fn = os.path.join(a.out, f"curve_k{a.k}_Q{a.Q}.json")
    json.dump({"k": a.k, "Q": a.Q, "episodes": a.episodes, "rho": a.rho, "rec": rec}, open(fn, "w"))
    xs = [d["ep"] for d in rec]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(xs, [d["mean"] for d in rec], "o-", label="mean ratio")
    ax[0].plot(xs, [d["max"] for d in rec], "s-", label="max ratio")
    ax[0].axhline(1.0, color="goldenrod", ls="--", lw=1); ax[0].set_ylim(0, 1.05)
    ax[0].set_xlabel("training episode"); ax[0].set_ylabel("hardened Phi/Phi*"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[0].set_title(f"k={a.k}: climb of the distribution")
    ax[1].plot(xs, [d["optfrac"] for d in rec], "o-", color="crimson", label="optfrac (>=.99)")
    ax[1].plot(xs, [d["ge95"] for d in rec], "^-", color="orange", label=">=.95")
    ax[1].set_xlabel("training episode"); ax[1].set_ylabel("fraction of pool"); ax[1].legend(); ax[1].grid(alpha=.3)
    ax[1].set_title(f"k={a.k}: optimum-rate vs episodes")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, f"curve_k{a.k}_Q{a.Q}.png"), dpi=130)
    print(f"saved curve_k{a.k}_Q{a.Q}.json/.png")

if __name__ == "__main__":
    main()
