"""Individual PSGA (MAC-REINFORCE) behavior: the learned policy climbs trap-to-trap.

NO HARDENING. Hardening (argmax) belongs to the meta-algorithm's candidate extraction,
not to PSGA. To depict PSGA's own behavior we track the potential of the LEARNED POLICY
pi itself -- the actual stochastic policy, measured WITHOUT the alpha-greedy uniform
mixing. (alpha-greedy is only a training-time variance-control device; pi is what PSGA
optimizes.) So this is neither the alpha-biased exploring potential nor the argmax: it is
Phi(pi), estimated by a large-batch rollout with exploration temporarily set to 0.

Because pi is uncapped, as it concentrates on the optimal joint actions Phi(pi) -> 1.0,
while a trapped run concentrates on a suboptimal profile and plateaus at that trap's true
level -- the full separation, no exploration ceiling. Basin escapes are read off this
curve (plateau-to-plateau jumps). Training itself still uses alpha=0.1 throughout.

Two panels, trap-prone init (1,2,2): LEFT fixed batch (method); RIGHT gradual batch 4->128
(food for thought). A few cherry-picked exemplar seeds per panel make the behavior legible;
the measured fraction reaching the optimum is printed in the title.
"""
import sys, os, pickle, random as pyrandom
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "claude_parallelized"))
import numpy as np, torch
from concurrent.futures import ProcessPoolExecutor
from meta_algorithm import make_env, randomize, optimal_phi, S, A, T, GAMMA
from vectorized_train import train_vectorized, vec_rollout, disc_returns

INIT = (1, 2, 2)
EPISODES, CHUNK, BATCH, LR, H, EVAL_B = 2000, 20, 32, 1e-3, 1, 512
SEARCH = 48                                     # seeds searched per schedule
WIN_C, RISE, FLAT, REACH = 6, 0.05, 0.02, 0.95  # plateau-jump detection (chunks); reach-optimum threshold
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_meta", "psga_data.pkl")


def policy_potential(pols, opt):
    """Phi(pi)/Phi* : potential of the LEARNED policy with exploration turned OFF (no alpha mix)."""
    saved = [p.exploration_rate for p in pols]
    for p in pols:
        p.exploration_rate = 0.0
    with torch.no_grad():
        _, _, pots = vec_rollout(pols, INIT, S, A, H, T, EVAL_B)
        val = disc_returns(pots, GAMMA)[0].mean().item()
    for p, s in zip(pols, saved):
        p.exploration_rate = s
    return val / opt


def run_gradual_chunk(pols, n, done):
    b = min(128, 4 * 2 ** (done // 20))         # batch doubles 4->128 every 20 eps
    train_vectorized(pols, INIT, n, b, S, A, H, T, GAMMA, LR, True)


def one_run(args):
    seed, mode, opt = args
    pyrandom.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    ecg = make_env(INIT); randomize(ecg)
    pols = [ag.policy_func for ag in ecg.agents]
    traj, done = [], 0
    while done < EPISODES:
        n = min(CHUNK, EPISODES - done)
        if mode == "gradual":
            run_gradual_chunk(pols, n, done)
        else:
            train_vectorized(pols, INIT, n, BATCH, S, A, H, T, GAMMA, LR, True)
        done += n
        traj.append(policy_potential(pols, opt))    # read-only Phi(pi) probe (no alpha)
    return seed, np.array(traj)


def smooth(x, w):
    if w <= 1 or len(x) < w:
        return x
    pad = w // 2
    return np.convolve(np.pad(x, pad, mode="edge"), np.ones(w) / w, mode="valid")[:len(x)]


def climbs(tr):
    """Chunk indices of plateau-to-plateau RISES, AFTER the initial warm-up into basin #1."""
    n = len(tr); i0 = 0
    while i0 < n - WIN_C and tr[i0 + WIN_C] - tr[i0] >= FLAT:    # skip warm-up climb
        i0 += 1
    marks, i = [], i0
    while i < n - WIN_C:
        if tr[i + WIN_C] - tr[i] > RISE:
            seg = tr[i:i + WIN_C + 1]; marks.append(i + int(np.argmax(np.diff(seg))) + 1); i += WIN_C
        else:
            i += 1
    return marks


def pick(scored):
    """2 climbers (reach optimum, most visible climbs) + 1 staller (climbs but stays low)."""
    climbers = sorted([r for r in scored if r[3]],
                      key=lambda r: (len(r[2]), r[1][-1] - r[1][:8].min()), reverse=True)[:2]
    pool = [r for r in scored if not r[3]]
    stallers = sorted([r for r in pool if len(r[2]) >= 1] or pool,
                      key=lambda r: (len(r[2]) >= 1, -r[1][-3:].mean()), reverse=True)[:1]
    return climbers + stallers


def generate_data():
    """Train all seeds (fixed batch) and return the raw Phi(pi) trajectories (the slow part)."""
    opt = optimal_phi(INIT)
    print(f"init {INIT} | optimal Phi = {opt:.2f} | searching {SEARCH} seeds", flush=True)
    tasks = [(s, "fixed", opt) for s in range(SEARCH)]
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1)) as ex:
        results = list(ex.map(one_run, tasks))
    data = {"opt": opt, "init": INIT, "fixed": [(seed, traj) for seed, traj in results],
            "params": dict(EPISODES=EPISODES, CHUNK=CHUNK, BATCH=BATCH, LR=LR, H=H,
                           EVAL_B=EVAL_B, SEARCH=SEARCH, EXPLORE=0.1)}
    return data


def plot(data):
    """Cheap: smooth, detect climbs, cherry-pick exemplars, render. No training here."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ch = data["params"]["CHUNK"]
    fig, ax = plt.subplots(figsize=(8, 5))
    palette = ["tab:green", "tab:blue", "tab:red"]
    scored = [(seed, sm, climbs(sm), sm[-3:].mean() >= REACH)
              for seed, traj in data["fixed"] for sm in [smooth(traj, 3)]]
    rate = np.mean([r[3] for r in scored])
    chosen = pick(scored)
    xmax = 1
    for c, (seed, sm, marks, reached) in zip(palette, chosen):
        x = (np.arange(len(sm)) + 1) * ch
        tag = "reaches optimum" if reached else "stalls at a trap"
        ax.plot(x, sm, color=c, lw=2.0, alpha=0.9, label=f"{tag} ({len(marks)} escapes)")
        for k in marks:
            ax.plot((k + 1) * ch, sm[k], marker="*", ms=13, color="black",
                    markeredgecolor="white", markeredgewidth=0.7, zorder=6)
            xmax = max(xmax, (k + 1) * ch)
    ax.plot([], [], marker="*", ms=12, color="black", ls="", label="basin escape (plateau jump)")
    ax.axhline(1.0, ls="--", lw=0.8, color="gray", alpha=0.7)
    ax.text(20, 1.012, "global optimum", fontsize=8, color="gray")
    ax.set_xlabel("PSGA episode")
    ax.set_ylabel(r"learned-policy potential  $\bar\Phi(\pi)/\bar\Phi^\star$  (no $\alpha$ mixing)")
    ax.set_title(r"Individual PSGA runs climb trap-to-trap via $\alpha$-greedy exploration"
                 f"\ninit {data['init']};  reach optimum: {rate:.0%} of {len(data['fixed'])} seeds", fontsize=10)
    ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.1); ax.set_xlim(0, min(data["params"]["EPISODES"], int(xmax * 1.3) + 150))
    out = "results_meta/fig_psga.png"; plt.tight_layout(); plt.savefig(out, dpi=150)
    print(f"reach {rate:.0%}/{len(data['fixed'])} | exemplar seeds {[r[0] for r in chosen]} | saved {out}", flush=True)


if __name__ == "__main__":
    force = "--train" in sys.argv                   # `--train` re-runs the search; otherwise use the cache
    if os.path.exists(DATA) and not force:
        data = pickle.load(open(DATA, "rb"))
        print(f"loaded cached trajectories from {DATA} (use --train to regenerate)", flush=True)
    else:
        data = generate_data()
        os.makedirs(os.path.dirname(DATA), exist_ok=True)
        pickle.dump(data, open(DATA, "wb"))
        print(f"saved trajectories to {DATA}", flush=True)
    plot(data)
