"""Learning-curve probe: train a few drone candidates and record hardened Phi/Phi*
across training episodes, to see WHERE the potential plateaus (i.e. how many episodes
are actually needed). Reuses the tested drone_train machinery (q=1 single system).

Usage:
  python remote/curve_probe.py --seeds 5 --episodes 1000 --batch 64 --H 1 --eval-every 20
Outputs: results_remote/curve_probe.png  + a printed table of final/plateau ratios.
"""
import os
for _v in ["OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"]:
    os.environ[_v] = "1"
import sys, time, argparse
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
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

SCHED = [4, 8, 16, 32, 64, 128]

def train_curve(n, H, episodes, batch, lr, alpha, seed, eval_every, star, eval_mode, eval_batch):
    pyrandom.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    strides_t, S = dt.strides_for(H)
    P = torch.nn.Parameter(torch.distributions.Dirichlet(torch.ones(dg.NODES)).sample((n, S)))
    opt = torch.optim.SGD([P], lr=lr)
    sched = (batch == "sched")
    def batch_at(ep):
        return SCHED[min(len(SCHED) - 1, ep * len(SCHED) // episodes)] if sched else int(batch)
    def evalphi():
        # hard: argmax (deterministic). soft: the learned policy pi itself, alpha=0
        # (NO exploration floor), expected over many stochastic rollouts.
        with torch.no_grad():
            if eval_mode == "hard":
                _, _, pots = dt.rollout(P, n, H, 1, 0.0, strides_t, greedy=True)
            else:
                _, _, pots = dt.rollout(P, n, H, eval_batch, 0.0, strides_t, greedy=False)
            return float(disc_returns(pots.detach(), dg.GAMMA)[0].mean()) / star
    nbad = 0
    xs, ys = [0], [evalphi()]
    for ep in range(1, episodes + 1):
        b = batch_at(ep)
        logps, rews, _ = dt.rollout(P, n, H, b, alpha, strides_t)
        rets = disc_returns(rews.detach(), dg.GAMMA); adv = rets - rets.mean(2, keepdim=True)
        loss = -(logps * adv).sum() / b
        opt.zero_grad(); loss.backward()
        # --- divergence guard: skip non-finite grads, clip the rest (prevents NaN in P
        #     -> Categorical(NaN) -> native SIGILL). Then re-sanitize P onto the simplex.
        if P.grad is None or not torch.isfinite(P.grad).all():
            nbad += 1; opt.zero_grad()
        else:
            torch.nn.utils.clip_grad_norm_([P], 10.0); opt.step()
        with torch.no_grad():
            Pn = project_onto_simplex(P.view(n * S, dg.NODES)).view(n, S, dg.NODES)
            Pn = torch.nan_to_num(Pn, nan=1.0 / dg.NODES).clamp(min=0)
            P.copy_(Pn / Pn.sum(-1, keepdim=True))
        if ep % eval_every == 0:
            xs.append(ep); ys.append(evalphi())
    return xs, ys, nbad

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed-base", type=int, default=0, help="first seed index (to isolate a specific seed)")
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--batch", type=str, default="64", help="fixed int (e.g. 64, 128) or 'sched'")
    ap.add_argument("--H", type=int, default=1)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--eval-mode", choices=["soft", "hard"], default="soft",
                    help="soft = expected Phi of the learned policy pi (alpha=0, stochastic, averaged); hard = argmax")
    ap.add_argument("--eval-batch", type=int, default=512, help="# rollouts to average for soft eval")
    ap.add_argument("--out", type=str, default="results_remote/curve_probe.png")
    a = ap.parse_args()
    n = 4
    star, _ = vi_vec(n, tuple([(0, dg.B)] * n))
    print(f"Phi*(VI n=4) = {star:.3f}   | {a.seeds} seeds x {a.episodes} ep, H={a.H}, batch {a.batch}, "
          f"train-alpha={a.alpha}, eval={a.eval_mode}(alpha=0)\n", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    plt.figure(figsize=(8.5, 5.2)); t0 = time.time(); finals = []
    for s in range(a.seed_base, a.seed_base + a.seeds):
        xs, ys, nbad = train_curve(n, a.H, a.episodes, a.batch, a.lr, a.alpha, s, a.eval_every, star, a.eval_mode, a.eval_batch)
        finals.append(ys[-1])
        # episode at which it first reaches 99% of its own final value (plateau onset)
        thr = 0.99 * ys[-1]; plat = next((xs[i] for i in range(len(ys)) if ys[i] >= thr), xs[-1])
        plt.plot(xs, ys, marker='.', ms=3, lw=1.1, label=f"seed {s}: final {ys[-1]:.3f}, plateau@{plat}")
        print(f"seed {s}: final={ys[-1]:.4f}  max={max(ys):.4f}  plateau_onset~{plat}ep  skipped_bad_grad={nbad}  ({time.time()-t0:.0f}s)", flush=True)
    plt.axhline(1.0, color='k', ls='--', lw=0.8, label='Phi* (optimum)')
    elabel = "soft-policy (alpha=0)  Phi / Phi*" if a.eval_mode == "soft" else "hardened  Phi / Phi*"
    plt.xlabel("training episode"); plt.ylabel(elabel)
    plt.title(f"Drone MAC-REINFORCE: {a.eval_mode}-policy potential (n=4, H={a.H}, batch {a.batch})")
    plt.legend(fontsize=7, loc='lower right'); plt.grid(alpha=0.3); plt.ylim(0, 1.05)
    plt.tight_layout(); plt.savefig(a.out, dpi=130)
    print(f"\nsaved {a.out}\nfinals: mean={np.mean(finals):.3f}  best={max(finals):.3f}  worst={min(finals):.3f}")

if __name__ == "__main__":
    main()
