"""Pool-based selection demo of the meta-algorithm (cheap, no retraining).

The existing 1000-seed run is a sample of REAL MAC-REINFORCE hardened policies
(~37% optimal, bimodal). For a target init we load the policies trained from that
init, evaluate them (Gbar, [Ubar_i], Phi) at the init, and run the distributed
accept rule over a stream of these real candidates. Shows the SELECTION mechanism
concentrate the incumbent on the optimum (ratio->1) and how v^beta(Pi*) grows as
beta->0. Decoupled from the (expensive) PSGA candidate generation.
"""
import sys, os, glob, csv, ast, random as pyrandom
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "claude_parallelized"))
import numpy as np, torch
from meta_algorithm import make_env, eval_hardened, accepts, optimal_phi, N

RUN = ("claude_parallelized/simulation_results/"
       "classic_reinforce_gradual_batchsize_direct_parameterization_w_10%_exploration_20251221_133140")

def run_id_to_init():
    m = {}
    with open(os.path.join(RUN, "results_summary.csv")) as f:
        for r in csv.DictReader(f):
            m[r["run_id"]] = tuple(ast.literal_eval(r["config_init_states_tuple"]))
    return m

def load_pool(target_init):
    """Eval every policy TRAINED FROM target_init, at target_init -> list of (Gbar,[Ubar],Phi)."""
    target = tuple(target_init)
    rid2init = run_id_to_init()
    ecg = make_env(target)
    pool = []
    for rid, init in rid2init.items():
        if init != target:
            continue
        mdir = os.path.join(RUN, "models", rid)
        if not all(os.path.exists(os.path.join(mdir, f"agent_{i}_policy.pth")) for i in range(N)):
            continue
        for i, ag in enumerate(ecg.agents):
            sd = torch.load(os.path.join(mdir, f"agent_{i}_policy.pth"), map_location="cpu")
            ag.policy_func.load_state_dict(sd)
        pool.append(eval_hardened(ecg))
    return pool

def build_pool(target_init):
    pool = load_pool(target_init)
    opt = optimal_phi(target_init)
    frac_opt = float(np.mean([s[2] / opt >= 0.99 for s in pool]))
    return pool, opt, frac_opt

def select_nu(pool, opt, beta, K, seed, burn_in=50, reduced=True):
    """Fraction of (post burn-in) epochs the incumbent sits at the optimum."""
    pyrandom.seed(seed); np.random.seed(seed)
    inc = pool[pyrandom.randrange(len(pool))]
    at_opt = []
    for k in range(K):
        cand = pool[pyrandom.randrange(len(pool))]
        if accepts(cand, inc, beta, reduced):
            inc = cand
        if k >= burn_in:
            at_opt.append(inc[2] / opt >= 0.99)
    return float(np.mean(at_opt))


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INITS = [(1, 2, 2), (0, 2, 2), (2, 2, 2), (1, 1, 1)]   # trap inits, varying pool@opt
    BETAS = [1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01]
    SEEDS = list(range(20)); K = 600

    plt.figure(figsize=(7, 5))
    print(f"{'init':>10} {'pool':>5} {'pool@opt':>9} | nu^beta(Pi*) across beta", flush=True)
    for init in INITS:
        pool, opt, frac_opt = build_pool(init)
        means, stds = [], []
        for b in BETAS:
            nus = [select_nu(pool, opt, b, K, seed=s) for s in SEEDS]
            means.append(np.mean(nus)); stds.append(np.std(nus))
        lbl = f"init {init} (pool@opt={frac_opt:.2f}, n={len(pool)})"
        plt.errorbar(BETAS, means, yerr=stds, marker="o", capsize=3, label=lbl)
        plt.axhline(frac_opt, ls=":", lw=0.8, alpha=0.5)   # plain-PG baseline (no selection)
        print(f"{str(init):>10} {len(pool):5d} {frac_opt:9.2f} | "
              + " ".join(f"{m:.2f}" for m in means), flush=True)

    plt.xscale("log"); plt.gca().invert_xaxis()   # beta -> 0 to the right
    plt.xlabel(r"temperature $\beta$  (→ 0 to the right)")
    plt.ylabel(r"$\nu^\beta(\Pi^\star)$  (fraction of time at optimum)")
    plt.title("Nash-hopping selection: concentration on the optimum as $\\beta\\to0$")
    plt.ylim(0, 1.02); plt.grid(alpha=0.3); plt.legend(fontsize=8)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig4_nu_beta.png")
    plt.tight_layout(); plt.savefig(out, dpi=150)
    print(f"\nsaved figure: {out}", flush=True)
