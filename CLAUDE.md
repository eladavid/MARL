# CLAUDE.md — MARL / MAC-REINFORCE experiment code

Onboarding for agents working in this repo. It backs the **MAC-REINFORCE** paper
(cooperative MARL; finite-horizon Markov potential game). The code trains
independent per-agent policy gradient on a toy **congestion game** and measures
convergence, Nash, and optimality.

> Status: research code — works, but messy. A refactor is planned. Prefer small,
> verified changes; don't assume tidy structure.

## Branches
- **`feature/projected_gd_with_batch_variance_control`** ← the paper-faithful branch. Use this.
  Implements direct parameterization + **projected** gradient ascent (simplex projection)
  + **α-greedy exploration** (the "batch variance control"). All notes below describe it.
- `main` / `master`, `feature/mac_reinforce_v0`, `feature/large_scale_sim`, `feature/entropy_reg` — older / alternates.

## Git / SSH auth (remote is `git@github.com:eladavid/MARL.git`, SSH)
The remote needs the SSH key unlocked. In a human's interactive Git Bash, run **`ghssh`** once per
session (function in `~/.bashrc`: starts ssh-agent, adds `~/.ssh/id_rsa` — no passphrase, tests GitHub).
**For an agent (Claude): `ssh-agent` env does NOT persist across separate tool calls**, so chain the
unlock into the *same* command as the git op:
```bash
eval "$(ssh-agent -s)" >/dev/null 2>&1 && \
  ssh-add /c/Users/eladdavid1/.ssh/id_rsa < /dev/null >/dev/null 2>&1 && \
  git pull origin feature/projected_gd_with_batch_variance_control   # (or push/fetch)
```

## Quick start
```bash
# from repo root (so `congestion_game` and `claude_parallelized` both import)
python3 run_single.py          # one run, custom config, seed 4146964028  (see below)
```
`run_single.py` (repo root) is a minimal **single-run** harness — no parallel pool,
no sweep. It calls `SimulationManager.run_single_simulation(cfg)` directly and dumps a
potential curve to `single_run_results/`. Use it to sanity-check the pipeline.

Full experiment sweeps (heavy — up to 1000 seeds):
```bash
cd claude_parallelized
python3 experiment_configs.py custom        # the paper config, 1000-seed sweep
python3 experiment_configs.py quick_test --dry-run
```

### Import-path gotcha
`parallel_simulation.py` lives in `claude_parallelized/` but imports `from congestion_game....`,
which lives at the **repo root**. So you need BOTH on `sys.path`. `run_single.py` does this
explicitly; PyCharm does it via the project root. Running a bare script from inside
`claude_parallelized/` will fail to find `congestion_game` unless the root is on `PYTHONPATH`.

## Architecture

### `congestion_game/` — the core environment + learning primitives
| file | what |
|---|---|
| `episodic_agent.py` | `EpisodicAgent`: holds a policy (`policy_func`), `init_state`, history buffer; `act()` samples from `policy_func(state)`; `get_augmented_state(H)` builds the history-buffered state. |
| `episodic_congestion_game.py` | `EpisodicCongestionGame`: the env. `step()` → all agents act, compute `g + u_i`, **deterministic transition `s_i' = a_i`** (`update_states`); `do_episode()` rolls `T` steps; `check_if_nash_eq()` brute-forces NE by enumerating each agent's deterministic policy maps. |
| `policies.py` | **`DirectTabularPolicy`** is the paper's policy (see code↔paper map). Also MLP/embedding policies (unused in the paper config) and `project_onto_simplex()`. |
| `reward_functions.py` | `g_func` (global), `make_u_i` (per-agent private), `make_potential_func` (Φ = g + Σuᵢ). The toy game's reward landscape. |
| `utils.py` | `evaluate_policy`, `compute_discounted_returns`, `freeze_joint_policy` (hashable policy key for Nash caching), `visualize_joint_mdp`. |
| `optimizers.py`, `simulation_runner.py`, `analysis.ipynb` | misc / older driver / exploratory notebook. |

### `claude_parallelized/` — orchestration, analysis, Nash
| file | what |
|---|---|
| `parallel_simulation.py` | **Main engine.** `SimulationConfig` (dataclass of all knobs), `SimulationManager` (run single / parallel, save configs/results/models/plots), `train()` (the REINFORCE loop), `find_joint_optimum()` (value iteration baseline), `run_single_simulation()`. |
| `experiment_configs.py` | Predefined experiments. **`create_my_custom_experiment()` is the paper config.** CLI: `python experiment_configs.py {custom,baseline,quick_test,...}`. |
| `nash_checker.py` | Standalone / post-hoc Nash verification. |
| `results_analyzer.py`, `results_analyzer_by_state.py` | Build the comparison plots (potential evolution, optimality histogram, learning curves, trajectory) and the per-initial-state breakdowns (gap / nash-rate / final-potential by state). |
| `simulation_results/<exp>_<timestamp>/` | All outputs: `configs/`, `results/` (pkl + summary json), `models/`, `comparison_plots/`, `comparison_plots_by_state/`, `aggregated_results.json`, `analysis_report.txt`. |

## Code ↔ paper map (important)
The paper's MAC-REINFORCE is implemented as:
- **Direct parameterization** — `DirectTabularPolicy.probs_table` is an `nn.Parameter` that **IS** `π(a|s)` (not logits). Shape `(state_vocab…, num_actions)`.
- **Projected gradient ascent** — SGD step on the REINFORCE loss, then `project_parameters_onto_simplex()` (Euclidean projection of each `π(·|s)` row onto Δ) called after `optimizer.step()` in `train()`. This is the paper's `Proj_Δ`.
- **α-greedy exploration** — `DirectTabularPolicy.forward`, last line:
  `probs = (1-α)·probs + α·(1/|A|)`, with `α = exploration_rate = 0.1`. This is the paper's `π^α`. It (1) bounds `π ≥ α/|A| > 0` → the **bounded-variance estimator `σ² ≤ T⁴R²|A|²/α²`** (note the `1/α²`); (2) guarantees sufficient exploration; (3) **biases the fixed point** (hardened argmax of an α-perturbed policy ≠ exact NE) — empirically: Nash rate ≈ 96% with exploration vs 100% without.
- **REINFORCE + baseline** — `train()`: per-agent score × discounted return, mean-baseline; `total_loss = Σ agents` then independent SGD optimizers (one per agent) — the "blind coordination" (independent updates = joint potential ascent under fixed `s₀`).
- **Deterministic decoupled dynamics** — `update_states`: `new_state = actions[i]`, i.e. `s_i' = a_i`. Matches `P(s_i'|s_i,a_i)=I_{a_i→s_i'}`.
- **History buffer** — `history_len` (`H`); `get_augmented_state(H)` concatenates the last `H` states. `H=1` in the paper config.
- **Meta-algorithm (stochastic Nash hopping)** — **NOT yet implemented here.** This is the planned next addition (produces the "trap-collapsed" optimality panel).

## The toy congestion game (reward landscape)
- 3 agents, 3 states, 3 actions. State `s_i∈{0,1,2}`, action `a_i∈{0,1,2}`, transition `s_i'=a_i`.
- `g_func(a)`: **10** if joint action is `[0,1,2]` or `[0,2,1]` (the coordinated optima), else a diversity/congestion term `1 − Σ(count_a/n)²`.
- `u_i(s_i,a_i)`: if `s_i=0`, reward action near 0 (`3(1−|s_i−a_i|/3)`); else reward action far from `s_i` (`3|s_i−a_i|/3`). The private pull = the cooperative trap.
- `Φ = g + Σ u_i`. Optimal joint policy found by value iteration (`find_joint_optimum`).

## SimulationConfig knobs (paper "custom" values)
`num_agents=3, state_dim=3, action_dim=3, history_len=1, episode_len=16, gamma=0.99,`
`batch_size=4` (auto-**doubles every 20 episodes up to 128** in `train()`), `num_episodes=1500,`
`use_episodic_freeze=False, use_baseline=True, learning_rate=1e-3,`
`init_states_tuple` (per-seed random if None), `check_nash_equilibrium` (expensive brute-force).
Exploration rate (`α=0.1`) is **hardcoded** in `DirectTabularPolicy`, not in the config.

## Gotchas
- **Windows-isms**: `set_process_affinity_safe` (psutil affinity) — harmless on mac/linux but written for Windows.
- **Gradual batch size**: `train()` mutates `batch_size` (×2 every 20 eps, cap 128). Most of a 1500-ep run is at batch 128, so runs are slow (~25 min single, ~1.5h × parallel for the sweep).
- **Nash check** enumerates every deterministic policy map per agent — only feasible for tiny `|S|,|A|,H`; cached to `*_nash_bool_dict.pkl`.
- **`run_single_simulation` does not save to disk** (only `run_parallel_simulations` does); `run_single.py` saves a JSON itself.
- Exact-optimum runs reach `gap = 0`; trap runs plateau (~0.57 of optimal). Outcome is **initial-state and seed dependent** (bimodal optimality histogram).

## Where to look for results / patterns
The most paper-faithful run is
`claude_parallelized/simulation_results/classic_reinforce_gradual_batchsize_direct_parameterization_w_10%_exploration_20251221_133140/`
(1500 ep, α=0.1, 432 seeds): `comparison_plots/potential_comparison.png`,
`optimality_gap_histogram.png`, `analysis_report.txt`, `nash_equilibrium_summary.json`.
