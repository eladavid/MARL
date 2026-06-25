"""Remote analysis: merge worker parts, run the VALIDATED meta-algorithm (Algorithm 2's
accept rule, meta_algorithm.accepts) over the pool, compute the strategy comparison, and
emit the figure + a text summary. Reuses tested code paths only.

Usage: python3 remote/analyze.py --dir results_remote [--rand 0.789]
"""
import os, sys, glob, pickle, argparse, random as pyrandom
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import meta_algorithm as M
LAB=(30/255,70/255,110/255); GOLD=(176/255,124/255,38/255); GRAY="0.55"; RED=(0.7,0.2,0.15); GREEN=(0.2,0.5,0.3)

def load(dirp, H, batch=None):
    stats=[]; ratios=[]; star=None
    pat = f"part_H{H}_*.pkl" if batch is None else f"part_H{H}_{batch}_*.pkl"
    for p in glob.glob(os.path.join(dirp, pat)):
        d=pickle.load(open(p,"rb")); stats+=d["stats"]; ratios+=d["ratios"]; star=d["star"]
    return stats, np.array(ratios), star

def meta_curve(stats, star, betas, epochs=4000, burn=400, seeds=6):
    M.N=4
    def chain(beta, seed):
        rng=pyrandom.Random(seed); inc=min(stats,key=lambda s:s[2]); v=[]
        for e in range(epochs):
            c=stats[rng.randrange(len(stats))]
            if M.accepts(c,inc,beta,reduced=True): inc=c
            if e>=burn: v.append(inc[2]/star)
        return np.mean(v)
    return [float(np.mean([chain(b,s) for s in range(seeds)])) for b in betas]

if __name__ == "__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--dir",default="results_remote")
    ap.add_argument("--rand",type=float,default=0.789); a=ap.parse_args()
    s1,r1,star=load(a.dir,1)                     # H=1 pool (all batch modes)
    s0,r0,star0=load(a.dir,0)                     # H=0 memoryless pool
    star=star or star0
    betas=[0.3,0.1,0.05,0.02,0.01,0.005]
    nu1=meta_curve(s1,star,betas) if s1 else []
    h0_best = float(r0.max()) if len(r0) else float("nan")
    h0_meta = meta_curve(s0,star,[0.005])[0] if s0 else float("nan")
    summary=[
        f"DRONE n=4 strategy comparison (Phi/Phi*), Phi*={star:.2f}",
        f"  H=0 memoryless ceiling (best of {len(r0)}):   {h0_best:.3f}   [+meta {h0_meta:.3f}]" if len(r0) else "  H=0: (no data)",
        f"  single PSGA H=1 (mean of {len(r1)}):           {r1.mean():.3f}  (worst {r1.min():.3f})" if len(r1) else "  H=1: (no data)",
        f"  random-deterministic search:                  {a.rand:.3f}",
        f"  MAC-REINFORCE + meta (cold beta):             {nu1[-1]:.3f}" if nu1 else "",
        f"  VI global optimum:                            1.000",
        f"  H=1 pool optima(>=0.99): {int((r1>=0.99).sum())} of {len(r1)}" if len(r1) else "",
    ]
    print("\n".join(x for x in summary if x), flush=True)
    open(os.path.join(a.dir,"SUMMARY.txt"),"w").write("\n".join(x for x in summary if x)+"\n")
    if len(r1):
        plt.rcParams.update({"axes.spines.top":False,"axes.spines.right":False})
        fig,(axL,axR)=plt.subplots(1,2,figsize=(8,3.3))
        axL.hist(r1,bins=np.linspace(0,1.02,40),color=LAB,alpha=0.8)
        if len(r0): axL.axvline(h0_best,color=GREEN,lw=1.8,ls=(0,(1,1)),label=f"H=0 ceiling ({h0_best:.2f})")
        axL.axvline(r1.mean(),color=GRAY,lw=1.5,label=f"single PSGA mean ({r1.mean():.2f})")
        axL.axvline(a.rand,color=RED,lw=1.5,ls="-.",label=f"random search ({a.rand:.2f})")
        axL.axvline(nu1[-1],color=LAB,lw=2.2,label=f"meta-algorithm ({nu1[-1]:.2f})")
        axL.axvline(1.0,color=GOLD,lw=1.6,ls="--",label=r"VI optimum (1.0)")
        axL.set_xlabel(r"$\Phi/\Phi^\star$ reached"); axL.set_ylabel(f"# of {len(r1)} runs")
        axL.set_title("Independent learning is a lottery (n=4)",fontsize=10); axL.legend(fontsize=7,loc="upper left")
        axR.plot(betas,nu1,"o-",color=LAB,lw=2,label=r"meta $\nu^\beta$")
        if len(r0): axR.axhline(h0_best,color=GREEN,lw=1.2,ls=(0,(1,1)),label="H=0 ceiling")
        axR.axhline(r1.mean(),ls=":",color=GRAY,lw=1.3,label="single PSGA mean")
        axR.axhline(a.rand,ls="-.",color=RED,lw=1.2,label="random search")
        axR.axhline(1.0,ls="--",color=GOLD,lw=1.4,label="VI optimum")
        axR.set_xscale("log"); axR.invert_xaxis(); axR.set_ylim(0,1.05)
        axR.set_xlabel(r"temperature $\beta$ (cooling $\rightarrow$)"); axR.set_ylabel(r"incumbent $\Phi/\Phi^\star$")
        axR.set_title("Meta extracts the best discoverable policy",fontsize=10); axR.legend(fontsize=7,loc="center left")
        fig.suptitle("Drone patrol: hard-to-find optimum; H=1 needed to represent it; meta dominates strategies",fontsize=9.5,y=1.02)
        fig.tight_layout(); fig.savefig(os.path.join(a.dir,"fig_drone_strategies.pdf"),bbox_inches="tight",pad_inches=0.03)
        print(f"\nsaved {a.dir}/fig_drone_strategies.pdf and SUMMARY.txt")
