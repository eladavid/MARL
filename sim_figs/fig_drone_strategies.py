"""Publication-grade drone-game figure (n=4). Panels show data; interpretation goes in
the LaTeX caption, not the figure. (a) distribution of the hardened return over many
independent MAC-REINFORCE runs; (b) steady-state concentration of the Nash-hopping
incumbent vs temperature, with baselines. Built from cached pools (no training)."""
import sys, os, glob, pickle, random as pyrandom
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import meta_algorithm as M

plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.titlesize": 11, "axes.labelsize": 11, "legend.fontsize": 8.5,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "lines.linewidth": 1.6,
})
LAB=(0.12,0.27,0.43); GOLD=(0.69,0.49,0.15); GRAY="0.5"; RED=(0.62,0.18,0.14); GREEN=(0.16,0.45,0.30)
HERE=os.path.dirname(os.path.abspath(__file__))

# --- load H=1 pool ---
stats=[]; ratios=[]; star=None
for p in glob.glob(os.path.join(HERE,'drone_pool_part_*.pkl'))+glob.glob(os.path.join(HERE,'drone_sched_part_*.pkl')):
    d=pickle.load(open(p,'rb')); stats+=d['stats']; ratios+=d['ratios']; star=d['star']
ratios=np.array(ratios); M.N=4
RAND=0.789                  # best of 1e6 random deterministic policies (memoryless-jump candidates)
H0=0.22                     # H=0 memoryless ceiling (measured)

# --- meta concentration with error bars ---
betas=[0.3,0.1,0.05,0.02,0.01,0.005]
def chain(beta,seed,epochs=4000,burn=400):
    rng=pyrandom.Random(seed); inc=min(stats,key=lambda s:s[2]); v=[]
    for e in range(epochs):
        c=stats[rng.randrange(len(stats))]
        if M.accepts(c,inc,beta,reduced=True): inc=c
        if e>=burn: v.append(inc[2]/star)
    return np.mean(v)
seeds=range(8)
nu_mean=np.array([np.mean([chain(b,s) for s in seeds]) for b in betas])
nu_std =np.array([np.std ([chain(b,s) for s in seeds]) for b in betas])

fig,(axA,axB)=plt.subplots(1,2,figsize=(7.2,2.9))

# (a) outcome distribution of independent runs
axA.hist(ratios,bins=np.linspace(0.2,1.0,33),color=LAB,alpha=0.85,edgecolor="white",linewidth=0.3)
axA.axvline(1.0,color=GOLD,lw=1.4,ls="--")
axA.text(0.985,axA.get_ylim()[1]*0.92,r"$\Phi^\star$",color=GOLD,ha="right",va="top",fontsize=10)
axA.set_xlabel(r"hardened return $\bar\Phi/\bar\Phi^\star$")
axA.set_ylabel(r"independent runs ($n{=}%d$)"%len(ratios))
axA.set_xlim(0.2,1.02)
axA.text(0.02,0.98,"(a)",transform=axA.transAxes,va="top",fontsize=11,fontweight="bold")

# (b) hopping concentration vs temperature, with baselines
axB.axhline(1.0,color=GOLD,lw=1.2,ls="--",label=r"global optimum $\bar\Phi^\star$")
axB.axhline(RAND,color=RED,lw=1.0,ls="-.",label="random search")
axB.axhline(ratios.mean(),color=GRAY,lw=1.0,ls=":",label="single run (mean)")
axB.axhline(H0,color=GREEN,lw=1.0,ls=(0,(1,1)),label="memoryless ($H{=}0$)")
axB.errorbar(betas,nu_mean,yerr=nu_std,marker="o",ms=4,color=LAB,capsize=2,label="Nash-hopping")
axB.set_xscale("log"); axB.invert_xaxis(); axB.set_ylim(0,1.08)
axB.set_xlabel(r"temperature $\beta$"); axB.set_ylabel(r"incumbent $\bar\Phi/\bar\Phi^\star$")
axB.set_xticks([0.1,0.01]); axB.set_xticklabels([r"$10^{-1}$",r"$10^{-2}$"])
axB.text(0.02,0.98,"(b)",transform=axB.transAxes,va="top",fontsize=11,fontweight="bold")
axB.legend(loc="lower left",frameon=False,handlelength=1.8)

fig.tight_layout(w_pad=1.5)
fig.savefig(os.path.join(HERE,"fig_drone_strategies.pdf"),bbox_inches="tight",pad_inches=0.02)
print(f"saved fig_drone_strategies.pdf | meta_cold={nu_mean[-1]:.3f}+-{nu_std[-1]:.3f} single={ratios.mean():.3f}")
