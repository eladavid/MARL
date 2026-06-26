"""Run the VALIDATED meta-algorithm (meta_algorithm.run_hopping, Algorithm 2) on the
graded k x k x k game, by ONLY swapping the module globals (N,S,A), g_func, make_u_i,
and optimal_phi. All decision logic (eval_hardened, accepts, two-stage run_hopping) is
the validated code, unchanged -- so the concentration result is trustworthy by reuse.

Concentration: for each beta, run K hopping epochs over SEEDS chains; nu^beta = fraction
of post-burn-in epochs the incumbent sits at the optimum. Stochastic stability => nu^beta
-> 1 as beta -> 0."""
import sys, os, pickle, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import meta_algorithm as M
from graded_game import closed_form, C as MOVE_C, W as COV_W

def patch_to_graded(k, rho):
    m = k
    M.N, M.S, M.A = k, m, m
    # R_MAX so that Gbar, Ubar land in (0,1) (same role as in the 3x3x3 toy): per-step
    # g<=W, per-agent u<=C -> per-step reward <= W + C; discounted over T, /(T*R_MAX).
    M.R_MAX = COV_W + MOVE_C

    def g_func(actions, num_agents):
        a = actions.tolist() if torch.is_tensor(actions) else list(actions)
        cov = len(set(int(x) for x in a))
        val = COV_W * ((1 - rho) * cov / m + rho * (1.0 if cov == m else 0.0))
        return torch.tensor(float(val))                 # episodic train stacks tensors

    def make_u_i(num_states):
        def u(s_i, a_i):
            return torch.tensor(float(MOVE_C * (1.0 if int(a_i) != int(s_i) else 0.0)))
        return u

    M.g_func = g_func
    M.make_u_i = make_u_i
    opt = closed_form(k, rho)
    M.optimal_phi = lambda init_states, _opt=opt: _opt          # known closed-form optimum
    return opt

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--rho", type=float, default=0.7)
    ap.add_argument("--K", type=int, default=120)          # hopping epochs per chain
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--burn", type=int, default=30)
    ap.add_argument("--betas", type=str, default="0.3,0.1,0.05,0.02,0.01")
    ap.add_argument("--out", type=str, default="")
    a = ap.parse_args()

    opt = patch_to_graded(a.k, a.rho)
    init = tuple([0] * a.k)
    betas = [float(b) for b in a.betas.split(",")]
    print(f"VALIDATED meta-algo on graded game: k={a.k} rho={a.rho} opt={opt:.2f} "
          f"K={a.K} seeds={a.seeds}\n", flush=True)
    nu, allratios = {}, {}
    for beta in betas:
        fr = []
        for s in range(a.seeds):
            ratios, _ = M.run_hopping(init, beta=beta, K=a.K, reduced=True, seed=s, verbose=False)
            fr.append(np.mean([r >= 0.99 for r in ratios[a.burn:]]))
        nu[beta] = float(np.mean(fr))
        allratios[beta] = fr
        print(f"  beta={beta:<5}: nu^beta(opt) = {nu[beta]:.3f}   (per-seed {[round(x,2) for x in fr]})",
              flush=True)
    out = a.out or f"sim_figs/meta_graded_k{a.k}.pkl"
    pickle.dump({"k": a.k, "rho": a.rho, "opt": opt, "betas": betas, "nu": nu,
                 "per_seed": allratios}, open(out, "wb"))
    print(f"\nsaved {out}")
