"""Analyze abstract-game scaling: for each agent-count k, merge worker parts, run the
VALIDATED meta-algorithm (meta_algorithm.accepts) over a beta grid -> concentration
nu^beta(k), and emit the scaling figure + summary.

The question this answers: does Nash-hopping STILL concentrate on the global optimum as
the number of agents grows (k=5,6,7,8), or does the mixing tax / vanishing optimum-rate
kill it? Both outcomes are informative.

Usage: python3 remote/abstract_analyze.py --dir results_abstract
"""
import os, sys, glob, pickle, argparse, random as pyrandom
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import meta_algorithm as M

def load_k(dirp, k):
    stats=[]; ratios=[]; opt=None
    for p in glob.glob(os.path.join(dirp, f"abs_k{k}_*.pkl")):
        d=pickle.load(open(p,"rb")); stats+=d["stats"]; ratios+=d["ratios"]; opt=d["opt"]
    return stats, np.array(ratios), opt

def chain(stats, beta, n, opt, epochs=8000, burn=2000, seed=0):
    M.N=n; rng=pyrandom.Random(seed); inc=min(stats,key=lambda s:s[2]); v=[]
    for e in range(epochs):
        c=stats[rng.randrange(len(stats))]
        if M.accepts(c,inc,beta,reduced=True): inc=c
        if e>=burn: v.append(inc[2]/opt>=0.99)
    return float(np.mean(v))

if __name__ == "__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--dir",default="results_abstract")
    ap.add_argument("--ks",default="5,6,7,8"); a=ap.parse_args()
    ks=[int(x) for x in a.ks.split(",")]
    betas=[0.3,0.1,0.05,0.02,0.01,0.005]
    plt.rcParams.update({"font.family":"serif","axes.spines.top":False,"axes.spines.right":False})
    fig,ax=plt.subplots(figsize=(4.2,3.2))
    cmap=plt.cm.viridis(np.linspace(0.15,0.85,len(ks)))
    lines=[]
    for ci,k in enumerate(ks):
        stats,ratios,opt=load_k(a.dir,k)
        if not stats: print(f"k={k}: no data"); continue
        p=float((ratios>=0.99).mean())
        nu=[np.mean([chain(stats,b,k,opt,seed=s) for s in range(4)]) for b in betas]
        ax.plot(betas,nu,"o-",color=cmap[ci],ms=3,label=r"$k=%d$ ($p=%.3f$)"%(k,p))
        lines.append((k,len(ratios),p,float(ratios.max()),nu[-1]))
        print(f"k={k}: Q={len(ratios)} pool optfrac(p)={p:.4f} best={ratios.max():.4f} "
              f"nu^beta(cold)={nu[-1]:.3f}", flush=True)
    ax.axhline(1.0,color="0.7",ls="--",lw=1)
    ax.set_xscale("log"); ax.invert_xaxis(); ax.set_ylim(0,1.05)
    ax.set_xlabel(r"temperature $\beta$"); ax.set_ylabel(r"time at $\Phi^\star$ ($\nu^\beta$)")
    ax.legend(fontsize=8,frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(a.dir,"fig_abstract_scaling.pdf"),bbox_inches="tight")
    with open(os.path.join(a.dir,"SUMMARY.txt"),"w") as f:
        f.write("k, pool_Q, opt_rate_p, best_ratio, nu_beta_cold\n")
        for r in lines: f.write(", ".join(str(x) for x in r)+"\n")
    print(f"\nsaved {a.dir}/fig_abstract_scaling.pdf + SUMMARY.txt")
