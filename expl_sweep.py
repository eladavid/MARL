"""Can exploration (alpha) / gradient variance (batch size) drive k=5 out of the
selection trap into Phi*? k=5, H=1 (representable), grid over alpha x batch.
Baseline was alpha=0.1, batch=64 -> mean 0.65, reach 0/6."""
from scaled_train import run
from scale_spike2 import g_gen, u_gen
from scale_spike import closed_form_optimum

k = 5
cf, _ = closed_form_optimum(k)
print(f"k={k}, H=1: exploration(alpha) x variance(batch) grid  (baseline a=0.1,b=64 -> 0.65, 0/6)\n")
for alpha in [0.1, 0.2, 0.35]:
    for batch in [16, 64]:
        run(k, k, g_gen, u_gen, cf, 1, 5, 2500, batch, 16, 0.99, 1e-3,
            f"a={alpha} b={batch}", alpha=alpha)
    print()
