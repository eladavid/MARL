"""Faithful Nash-hopping meta-algorithm experiment (paper Algorithm 2), built for
cluster execution. Two phases:

  generate : (EXPENSIVE, parallel) for each initial state, draw K random restarts;
             each produces a pair  jump = harden(theta_rand)  and
             cand = harden(MAC-REINFORCE(theta_rand)).  Cached to disk.
  select   : (CHEAP) replay the cached pairs through the two-stage accept rule for a
             grid of temperatures beta and many seeds -> nu^beta(Pi*); save CSV + figure.

Candidate generation is incumbent-independent, so parallel-generate + sequential-replay
== running Algorithm 2 one epoch at a time, in distribution.

Examples
--------
  # one command, modest local run:
  python run_meta_experiment.py all --inits 122 022 222 111 --K 200 \
        --sub-episodes 800 --sub-batch 64 --out results_meta

  # cluster: generate big once, then sweep cheaply
  python run_meta_experiment.py generate --inits 122 022 222 111 000 012 \
        --K 1000 --sub-episodes 1500 --sub-batch 64 --workers 32 --out results_meta
  python run_meta_experiment.py select   --inits 122 022 222 111 000 012 \
        --epochs 2000 --seeds 50 --out results_meta
"""
import sys, os, argparse, pickle, csv, random as pyrandom
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "claude_parallelized"))
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from meta_algorithm import accepts, optimal_phi
from meta_parallel import gen_candidate

BETAS = [1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01]


def _atomic_save(path, data):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(data, f)
    os.replace(tmp, path)           # atomic: never leaves a half-written file on interrupt


def _load_or_init(path, init):
    if os.path.exists(path):
        data = pickle.load(open(path, "rb"))
        if not isinstance(data.get("pairs"), dict):     # migrate old list format
            data["pairs"] = {i: p for i, p in enumerate(data["pairs"])}
        return data
    return {"init": init, "opt": optimal_phi(init), "pairs": {}}     # pairs keyed by seed


def parse_init(s):                       # "122" -> (1,2,2)
    return tuple(int(c) for c in s)


def two_stage_select(pairs, opt, beta, epochs, seed, burn_in, reduced=True):
    """Replay pre-generated (jump, cand) pairs through Algorithm 2's two-stage rule."""
    pyrandom.seed(seed); np.random.seed(seed)
    inc = pairs[pyrandom.randrange(len(pairs))][1]      # start from a PSGA candidate
    at_opt, early, psga = [], 0, 0
    for k in range(epochs):
        jump, cand = pairs[pyrandom.randrange(len(pairs))]
        if accepts(jump, inc, beta, reduced):           # stage 2: hardening early-stop
            inc = jump; early += 1
        else:                                           # stage 3/4: PSGA candidate
            psga += 1
            if accepts(cand, inc, beta, reduced):
                inc = cand
        if k >= burn_in:
            at_opt.append(inc[2] / opt >= 0.99)
    return float(np.mean(at_opt)), early / epochs, psga / epochs


def cmd_generate(a):
    """Incremental + resumable: each candidate saved as it completes; re-running
    skips already-done seeds and continues. Atomic writes survive interrupts."""
    os.makedirs(a.out, exist_ok=True)
    workers = a.workers or max(1, (os.cpu_count() or 2) - 1)
    for s_str in a.inits:
        init = parse_init(s_str)
        path = os.path.join(a.out, f"pairs_{s_str}.pkl")
        data = _load_or_init(path, init)
        data["sub_episodes"], data["sub_batch"] = a.sub_episodes, a.sub_batch
        opt = data["opt"]
        todo = [s for s in range(a.K) if s not in data["pairs"]]
        print(f"[generate] init {init}  opt={opt:.2f}  have {len(data['pairs'])}/{a.K}, "
              f"generating {len(todo)} more (ep={a.sub_episodes}, batch={a.sub_batch}, "
              f"workers={workers}, save_every={a.save_every})", flush=True)
        if not todo:
            print("           already complete — skipping.", flush=True); continue
        args = [(s, init, a.sub_episodes, a.sub_batch, a.lr) for s in todo]
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(gen_candidate, ar): ar[0] for ar in args}
            for i, fut in enumerate(as_completed(futs), 1):
                data["pairs"][futs[fut]] = fut.result()
                if i % a.save_every == 0 or i == len(todo):
                    _atomic_save(path, data)
                    cr = [c[2] / opt for _, c in data["pairs"].values()]
                    print(f"           [{len(data['pairs'])}/{a.K}] saved | "
                          f"PSGA @opt {np.mean([r>=0.99 for r in cr]):.2f} | mean {np.mean(cr):.3f}", flush=True)
        print(f"           done init {init}: {len(data['pairs'])} pairs at {path}", flush=True)


def cmd_select(a):
    rows = []
    for s in a.inits:
        p = os.path.join(a.out, f"pairs_{s}.pkl")
        if not os.path.exists(p):
            print(f"[select] missing {p} (run `generate` first) — skipping", flush=True); continue
        d = pickle.load(open(p, "rb")); init, opt = d["init"], d["opt"]
        pairs = list(d["pairs"].values()) if isinstance(d["pairs"], dict) else d["pairs"]
        pool_at_opt = float(np.mean([c[2] / opt >= 0.99 for _, c in pairs]))
        for b in BETAS:
            nus = [two_stage_select(pairs, opt, b, a.epochs, seed=i, burn_in=a.burn_in)[0]
                   for i in range(a.seeds)]
            rows.append(dict(init=s, beta=b, pool_at_opt=round(pool_at_opt, 4),
                             nu_mean=round(float(np.mean(nus)), 4), nu_std=round(float(np.std(nus)), 4)))
        print(f"[select] init {init}  pool@opt={pool_at_opt:.2f}  "
              + " ".join(f"{r['beta']}:{r['nu_mean']:.2f}" for r in rows if r['init'] == s), flush=True)

    with open(os.path.join(a.out, "nu_beta.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["init", "beta", "pool_at_opt", "nu_mean", "nu_std"])
        w.writeheader(); w.writerows(rows)

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 5))
        for s in sorted(set(r["init"] for r in rows)):
            rs = [r for r in rows if r["init"] == s]
            bs = [r["beta"] for r in rs]; m = [r["nu_mean"] for r in rs]; sd = [r["nu_std"] for r in rs]
            plt.errorbar(bs, m, yerr=sd, marker="o", capsize=3, label=f"init {parse_init(s)} (pool@opt={rs[0]['pool_at_opt']:.2f})")
            plt.axhline(rs[0]["pool_at_opt"], ls=":", lw=0.8, alpha=0.4)
        plt.xscale("log"); plt.gca().invert_xaxis(); plt.ylim(0, 1.02)
        plt.xlabel(r"temperature $\beta$  ($\to 0$ to the right)")
        plt.ylabel(r"$\nu^\beta(\Pi^\star)$"); plt.grid(alpha=0.3); plt.legend(fontsize=8)
        plt.title("Nash-hopping selection: concentration on the optimum")
        out = os.path.join(a.out, "fig4_nu_beta.png"); plt.tight_layout(); plt.savefig(out, dpi=150)
        print(f"[select] wrote {os.path.join(a.out,'nu_beta.csv')} and {out}", flush=True)
    except Exception as e:
        print(f"[select] CSV written; plot skipped ({e})", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["generate", "select", "all"])
    ap.add_argument("--inits", nargs="+", default=["122", "022", "222", "111"],
                    help='initial states as digit strings, e.g. 122 = (1,2,2)')
    ap.add_argument("--K", type=int, default=200, help="candidate pairs per init (generate)")
    ap.add_argument("--sub-episodes", dest="sub_episodes", type=int, default=800)
    ap.add_argument("--sub-batch", dest="sub_batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--save-every", dest="save_every", type=int, default=5,
                    help="checkpoint the pairs file every N completed candidates (resumable)")
    ap.add_argument("--epochs", type=int, default=2000, help="selection-chain length (select)")
    ap.add_argument("--seeds", type=int, default=50, help="selection seeds for error bars")
    ap.add_argument("--burn-in", dest="burn_in", type=int, default=200)
    ap.add_argument("--out", default="results_meta")
    a = ap.parse_args()
    if a.phase in ("generate", "all"): cmd_generate(a)
    if a.phase in ("select", "all"):   cmd_select(a)
