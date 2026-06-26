"""Collect all data for the toy-experiment simulation figures (graded game, rho=0.7).
Saves a single pickle sim_figs/sim_data.pkl consumed by the fig_*.py generators.
All runs at k<=5 for tractability. Run: python3 sim_data.py"""
import sys, os, time, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import scaled_vec as sv, scaled_pool as spool, scale_spike
from graded_game import make_g_torch, vi, closed_form, W, C
from scale_spike2 import u_gen
from required_buffer import required_buffer, states_from

RHO = 0.7
os.makedirs("sim_figs", exist_ok=True)
D = {"RHO": RHO, "W": W, "C": C}
t0 = time.time()

# ---- FIG1: optimum vs trap per-step values + optimal trajectory (k=4) ----
star4, traj4 = vi(4, RHO)
D["fig1"] = {"k": 4, "phistar_step": W + 4 * C, "trap_step": W,  # full coverage moving vs static
             "traj": [tuple(int(x) for x in a) for a in traj4],
             "phistar_disc": star4}
print(f"[fig1] k=4 opt traj collected ({time.time()-t0:.0f}s)", flush=True)

# ---- FIG2a: training reach vs buffer H (k=4, graded rho=0.7) ----
sv.g_gen = make_g_torch(RHO)  # patch (scaled_vec imports g_gen at call via scale_spike2; patch there)
import scale_spike2; scale_spike2.g_gen = make_g_torch(RHO)
gfn = make_g_torch(RHO)
cf4 = closed_form(4, RHO)
reach_vs_H = {}
for H in [0, 1, 2]:
    r = sv.run(4, 4, gfn, u_gen, cf4, H, 8, 5000, 32, 16, 0.99, 1e-3, f"k4H{H}")
    reach_vs_H[H] = r.tolist()
D["fig2a"] = {"k": 4, "reach_vs_H": reach_vs_H}
print(f"[fig2a] reach vs H done ({time.time()-t0:.0f}s)", flush=True)

# ---- FIG2b: required buffer H* (traced from optimum) vs worst-case bound k^(k-1) ----
reqH, bound = {}, {}
for k in [3, 4, 5]:
    star, traj = vi(k, RHO)
    Hs = required_buffer([tuple(int(x) for x in a) for a in traj], [0] * k, k)
    reqH[k] = max(h for h in Hs if h is not None)
    bound[k] = k ** (k - 1)
D["fig2b"] = {"reqH": reqH, "bound": bound}
print(f"[fig2b] required buffer: {reqH} vs bound {bound} ({time.time()-t0:.0f}s)", flush=True)

# ---- FIG3: pool reach distribution + hopping concentration (k=4 graded) ----
spool.g_gen = make_g_torch(RHO)
scale_spike.closed_form_optimum = lambda n, T=16, gamma=0.99: (closed_form(n, RHO), None)
spool.closed_form_optimum = scale_spike.closed_form_optimum
pool_ratios = spool.build_pool(4, 250, episodes=5000, batch=32, chunk=125, log=False)
D["fig3a"] = {"k": 4, "pool_ratios": pool_ratios.tolist(),
              "mean": float(pool_ratios.mean()), "popt": float((pool_ratios >= 0.99).mean())}
print(f"[fig3a] pool k=4: mean={pool_ratios.mean():.3f} popt={(pool_ratios>=0.99).mean():.3f} "
      f"({time.time()-t0:.0f}s)", flush=True)

# ---- FIG4b: reach (mean + p_opt) vs #agents at rho=0.7 ----
reach_vs_k = {}
for k in [3, 4, 5]:
    cf = closed_form(k, RHO)
    scale_spike.closed_form_optimum = lambda n, T=16, gamma=0.99, _c=cf: (_c, None)
    spool.closed_form_optimum = scale_spike.closed_form_optimum
    r = spool.build_pool(k, 120, episodes=5000, batch=32, chunk=120, log=False)
    reach_vs_k[k] = {"mean": float(r.mean()), "popt": float((r >= 0.99).mean()),
                     "ratios": r.tolist()}
    print(f"[fig4b] k={k}: mean={r.mean():.3f} popt={(r>=0.99).mean():.3f} ({time.time()-t0:.0f}s)", flush=True)
D["fig4b"] = reach_vs_k

# ---- FIG4a: brute-force VI cost vs per-agent table (analytic) ----
D["fig4a"] = {"k": list(range(2, 11)),
              "vi_cost": [(k ** k) ** 2 * 16 for k in range(2, 11)],
              "per_agent": [(k + 1) * k * k for k in range(2, 11)],  # [m+1]^H * m * A, H=1
              "vi_wall_k": 6}

pickle.dump(D, open("sim_figs/sim_data.pkl", "wb"))
print(f"\nSAVED sim_figs/sim_data.pkl  (total {time.time()-t0:.0f}s)")
