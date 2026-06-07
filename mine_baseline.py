"""Before-baseline: optimality of MAC-REINFORCE (H=1, hardened policy) restricted
to the 6 'trap' initial states, from the existing 1000-seed run. No new compute."""
import csv, os, ast, statistics

D = "claude_parallelized/simulation_results/classic_reinforce_gradual_batchsize_direct_parameterization_w_10%_exploration_20251221_133140"
TRAP = {(1,1,1),(2,2,2),(0,1,1),(0,2,2),(1,2,2),(2,1,1)}
OPT_EPS = 0.99      # ratio >= this counts as "reached optimum"
H0_CEIL = 0.88      # below this ~ at/under the raw-AD ceiling (deep trap)

rows = []
with open(os.path.join(D, "results_summary.csv")) as f:
    for r in csv.DictReader(f):
        init = tuple(ast.literal_eval(r["config_init_states_tuple"]))
        opt = float(r["optimal_potential"]); fin = float(r["final_potential"])
        if opt <= 0: continue
        rows.append((init, fin/opt))

trap_rows = [(i, ratio) for (i, ratio) in rows if i in TRAP]
print(f"total runs: {len(rows)} | runs at the 6 trap inits: {len(trap_rows)}\n")

def summarize(label, data):
    if not data: print(f"{label}: (none)"); return
    ratios = [x for _, x in data]
    opt_frac = sum(1 for x in ratios if x >= OPT_EPS) / len(ratios)
    trap_frac = sum(1 for x in ratios if x < H0_CEIL) / len(ratios)
    print(f"{label:>10} | n={len(ratios):3d} | mean ratio {statistics.mean(ratios):.3f} "
          f"| median {statistics.median(ratios):.3f} | @optimum {opt_frac:5.1%} | deep-trap(<{H0_CEIL}) {trap_frac:5.1%}")

print("=== per trap init ===")
for init in sorted(TRAP):
    summarize(str(init), [(i, x) for (i, x) in trap_rows if i == init])
print("\n=== aggregate over 6 trap inits ===")
summarize("TRAP", trap_rows)
print("\n=== aggregate over the 21 easy inits (sanity: should be ~all optimal) ===")
summarize("EASY", [(i, x) for (i, x) in rows if i not in TRAP])
