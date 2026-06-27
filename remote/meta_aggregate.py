"""Aggregate resilient meta-chain results (nu_b*_s*.json) -> nu^beta table + figure."""
import os, sys, glob, json, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_meta_drone")
    ap.add_argument("--fig", default="results_meta_drone/fig_meta_nu_beta.png")
    a = ap.parse_args()
    rows = [json.load(open(p)) for p in glob.glob(os.path.join(a.out, "nu_b*_s*.json"))]
    if not rows:
        print("no results"); return
    betas = sorted({r["beta"] for r in rows}, reverse=True)   # warm -> cold
    print(f"{'beta':>8} {'nu_mean':>8} {'nu_std':>7} {'best_final':>11} {'opt_reached':>12} {'seeds':>6}")
    xs, ys, es = [], [], []
    for b in betas:
        g = [r for r in rows if r["beta"] == b]
        nus = np.array([r["nu"] for r in g]); finals = np.array([r["final_incumbent"] for r in g])
        reached = sum(f >= 0.99 for f in finals)
        print(f"{b:>8.3f} {nus.mean():>8.4f} {nus.std():>7.4f} {finals.max():>11.4f} {reached:>8}/{len(g):<3} {len(g):>6}")
        xs.append(b); ys.append(nus.mean()); es.append(nus.std())
    plt.figure(figsize=(7, 5))
    plt.errorbar(xs, ys, yerr=es, fmt="o-", lw=2, capsize=3, label=r"meta $\nu^\beta$ (mean$\pm$std)")
    plt.axhline(1.0, color="goldenrod", ls="--", lw=1.2, label="VI optimum")
    plt.xscale("log"); plt.gca().invert_xaxis()
    plt.xlabel(r"temperature $\beta$ (cooling $\rightarrow$)"); plt.ylabel(r"incumbent $\Phi/\Phi^*$")
    plt.title("Drone meta-algorithm: nu^beta vs temperature (50k pool, p>0)")
    plt.ylim(0, 1.05); plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(a.fig, dpi=130)
    print(f"\nsaved {a.fig}")

if __name__ == "__main__":
    main()
