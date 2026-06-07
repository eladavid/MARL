"""Single MAC-REINFORCE run on the toy congestion game.

Reproduces the 'custom' experiment config (create_my_custom_experiment) for ONE
seed (4146964028 — the commented 'nice convergence' seed), via a direct call to
run_single_simulation (no parallel pool, no sweep). For familiarization only.
"""
import sys, os, json, random

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)                                   # so `congestion_game` resolves
sys.path.insert(0, os.path.join(ROOT, "claude_parallelized"))  # so `parallel_simulation` resolves

from parallel_simulation import SimulationConfig, SimulationManager, make_random_init_tuple

SEED = 4146964028

cfg = SimulationConfig(
    num_agents=3, state_dim=3, action_dim=3, history_len=1,
    episode_len=16, gamma=0.99,
    batch_size=4, num_episodes=1500,
    use_episodic_freeze=False, use_baseline=True, learning_rate=1e-3,
    init_states_tuple=None,
    experiment_name="single_custom_run", check_nash_equilibrium=False,
    seed=SEED,
)
# init states exactly as create_parameter_sweep_configs would set them
rng = random.Random(cfg.seed)
cfg.init_states_tuple = make_random_init_tuple(cfg.num_agents, cfg.state_dim, rng)
cfg.run_id = f"single_seed_{SEED}"
print("CONFIG:", cfg, flush=True)

mgr = SimulationManager(base_output_dir=os.path.join(ROOT, "single_run_results"))
results, agents = mgr.run_single_simulation(cfg)

pots = results.episode_potentials
opt = results.optimal_episode_discounted_potential
ach = results.argmax_episode_discounted_potential
print("\n===== RESULTS =====", flush=True)
print("init_states          :", cfg.init_states_tuple)
print("optimal disc. potent. :", round(opt, 4))
print("achieved disc. potent.:", round(ach, 4), f"({100*ach/opt:.1f}% of optimal)" if opt else "")
print("episodes recorded     :", len(pots))
if pots:
    print("potential first/mid/last:", round(pots[0],4), round(pots[len(pots)//2],4), round(pots[-1],4))
print("final trajectory      :", results.final_trajectory)
print("training_time (s)     :", round(results.training_time, 1))

out = os.path.join(ROOT, "single_run_results", "potential_curve.json")
with open(out, "w") as f:
    json.dump({
        "seed": SEED,
        "init_states": list(cfg.init_states_tuple),
        "optimal": opt, "achieved": ach,
        "episode_potentials": pots,
        "final_trajectory": [list(x) if isinstance(x, (list, tuple)) else x
                             for x in results.final_trajectory],
    }, f, indent=2)
print("saved:", out, flush=True)
