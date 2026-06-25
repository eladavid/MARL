"""Drone game (n=4): how hard the optimum is to find, and meta-algorithm vs other strategies.
Left: single-PSGA outcomes are a lottery (histogram), with strategy markers.
Right: meta-algorithm (validated accepts) concentrates the incumbent on the best
discoverable policy as beta cools, dominating single-run and random search.
The residual gap to the VI optimum quantifies how hard this realistic instance is."""
import sys, os, glob, pickle, random as pyrandom
sys.path.insert(0,'/Users/eladd/Downloads/CoMPG/MARL')
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import meta_algorithm as M
LAB=(30/255,70/255,110/255); GOLD=(176/255,124/255,38/255); GRAY="0.55"; RED=(0.7,0.2,0.15)
HERE=os.path.dirname(os.path.abspath(__file__))
stats=[]; ratios=[]; star=None
for p in glob.glob(os.path.join(HERE,'drone_pool_part_*.pkl'))+glob.glob(os.path.join(HERE,'drone_sched_part_*.pkl')):
    d=pickle.load(open(p,'rb')); stats+=d['stats']; ratios+=d['ratios']; star=d['star']
ratios=np.array(ratios); M.N=4
RAND=0.789  # best of 1M random-deterministic policies
def chain(beta,epochs=4000,burn=400,seed=0):
    rng=pyrandom.Random(seed); inc=min(stats,key=lambda s:s[2]); v=[]
    for e in range(epochs):
        c=stats[rng.randrange(len(stats))]
        if M.accepts(c,inc,beta,reduced=True): inc=c
        if e>=burn: v.append(inc[2]/star)
    return np.mean(v)
betas=[0.3,0.1,0.05,0.02,0.01,0.005]
nu=[np.mean([chain(b,seed=s) for s in range(6)]) for b in betas]
meta_cold=nu[-1]
plt.rcParams.update({"axes.spines.top":False,"axes.spines.right":False})
fig,(axL,axR)=plt.subplots(1,2,figsize=(8,3.3))
# LEFT: single-run distribution + strategy markers
axL.hist(ratios,bins=np.linspace(0,1.02,40),color=LAB,alpha=0.8)
axL.axvline(ratios.mean(),color=GRAY,lw=1.5,label=f"single PSGA mean ({ratios.mean():.2f})")
axL.axvline(RAND,color=RED,lw=1.5,ls="-.",label=f"random search ({RAND:.2f})")
axL.axvline(meta_cold,color=LAB,lw=2.2,label=f"meta-algorithm ({meta_cold:.2f})")
axL.axvline(1.0,color=GOLD,lw=1.6,ls="--",label=r"VI optimum $\Phi^\star$ (1.0)")
axL.set_xlabel(r"$\Phi/\Phi^\star$ reached"); axL.set_ylabel(f"# of {len(ratios)} independent runs")
axL.set_title("Independent learning is a lottery (n=4 drones)",fontsize=10)
axL.legend(fontsize=7.5,loc="upper left")
# RIGHT: meta concentration vs beta
axR.plot(betas,nu,"o-",color=LAB,lw=2,label=r"meta-algorithm $\nu^\beta$")
axR.axhline(ratios.mean(),ls=":",color=GRAY,lw=1.3,label="single PSGA mean")
axR.axhline(RAND,ls="-.",color=RED,lw=1.2,label="random search")
axR.axhline(1.0,ls="--",color=GOLD,lw=1.4,label=r"VI optimum")
axR.set_xscale("log"); axR.invert_xaxis(); axR.set_ylim(0,1.05)
axR.set_xlabel(r"temperature $\beta$ (cooling $\rightarrow$)")
axR.set_ylabel(r"incumbent $\Phi/\Phi^\star$")
axR.set_title("Meta-algorithm extracts the best discoverable policy",fontsize=10)
axR.legend(fontsize=7.5,loc="center left")
fig.suptitle(r"Drone patrol: optimum is 1 of $\sim$10$^{135}$ policies (0 found in 10$^6$ samples); meta-algorithm dominates other strategies",fontsize=9.5,y=1.02)
fig.tight_layout(); fig.savefig(os.path.join(HERE,"fig_drone_strategies.pdf"),bbox_inches="tight",pad_inches=0.03)
print(f"saved fig_drone_strategies.pdf | meta_cold={meta_cold:.3f} single={ratios.mean():.3f} rand={RAND}")
