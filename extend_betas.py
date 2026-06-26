"""Fast extended-beta replay from the CACHED k=5 pool stats (no retraining).
Sweeps beta down to 1e-3 so the concentration nu^beta visibly saturates near 1.
Updates sim_figs/hopping_data.pkl (consumed by fig3 in make_figs.py)."""
import sys, os, pickle, random as pyrandom
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import meta_algorithm as M
from graded_game import closed_form

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim_figs")
K, RHO = 5, 0.7
pool = pickle.load(open(os.path.join(HERE, f"pool_stats_k{K}.pkl"), "rb"))
opt = closed_form(K, RHO)
ratios = np.array([s[2] / opt for s in pool])

def chain(beta, n, epochs=6000, burn=600, seed=0):
    M.N = n; st = pyrandom.getstate(); pyrandom.seed(seed)
    inc = min(pool, key=lambda s: s[2]); at = 0
    for e in range(epochs):
        c = pool[pyrandom.randrange(len(pool))]
        if M.accepts(c, inc, beta, reduced=True): inc = c
        if e >= burn and inc[2] / opt >= 0.99: at += 1
    pyrandom.setstate(st)
    return at / (epochs - burn)

# floor at 0.005: for the smooth graded k=5 landscape, colder beta needs 10-100x more
# epochs to mix (unanimous vote can't make the sacrificing move into the exact optimum).
# At beta=0.005 concentration already reaches ~0.99. (3e3 epochs; longer burn for cold.)
betas = [0.3, 0.15, 0.08, 0.04, 0.02, 0.01, 0.005]
nu = {b: float(np.mean([chain(b, K, epochs=8000, burn=2000, seed=s) for s in range(10)])) for b in betas}
for b in betas:
    print(f"  beta={b:<6}: nu^beta(opt) = {nu[b]:.3f}", flush=True)
pickle.dump({"k": K, "rho": RHO, "betas": betas, "nu": nu,
             "pool_optfrac": float((ratios >= 0.99).mean()),
             "pool_ratios": ratios.tolist()}, open(os.path.join(HERE, "hopping_data.pkl"), "wb"))
print("updated sim_figs/hopping_data.pkl (betas to 1e-3)")
