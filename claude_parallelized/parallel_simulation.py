# parallel_simulation.py - Enhanced version with Nash equilibrium checking

import pickle
import random

import numpy as np
import psutil

import torch
import itertools
from tqdm import tqdm
import os
import pickle as pkl
import json
import time
import multiprocessing as mp
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
import logging

from congestion_game.episodic_agent import EpisodicAgent
from congestion_game.episodic_congestion_game import EpisodicCongestionGame
from congestion_game.policies import make_linear_softmax_policy, AgentPolicy, DiscreteStatePolicy, \
    DiscreteStatePolicyNoEmbeddings
from congestion_game.reward_functions import g_func, make_u_i, make_potential_func, make_random_g_func, \
    make_random_u_funcs
import torch.optim as optim
from congestion_game.utils import evaluate_policy, visualize_joint_mdp, compute_discounted_returns, freeze_joint_policy


def set_process_affinity_safe(excluded_cores=None, nice_value=5):
    """
    Safely sets CPU affinity on Windows without triggering WinError 87.
    Automatically respects processor groups.

    :param excluded_cores: list of logical CPU indexes to exclude (relative to current group)
    :param nice_value: optional process priority (Windows nice values: -20 to 19)
    """
    try:
        if excluded_cores is None:
            excluded_cores = [0, 1]

        p = psutil.Process(os.getpid())

        # Get the current allowed CPUs (this is the process's current processor group)
        current_affinity = p.cpu_affinity()
        group_max_cpu = max(current_affinity)

        # Keep only CPUs in this group
        allowed_cores = [cpu for cpu in current_affinity if cpu not in excluded_cores]

        if not allowed_cores:
            raise ValueError("No CPUs left after exclusion — check excluded_cores.")

        p.cpu_affinity(allowed_cores)

        # Set nice value (priority)
        try:
            p.nice(nice_value)
        except psutil.AccessDenied:
            print("Warning: Could not set process priority — need admin privileges.")

        print(f"[PID {p.pid}] Affinity set to {allowed_cores} (excluded {excluded_cores})")

    except Exception as e:
        print(f"Warning: Could not set process affinity: {e}")

@dataclass
class SimulationConfig:
    """Configuration for a single simulation run"""
    # Environment parameters
    num_agents: int = 3
    state_dim: int = 3
    action_dim: int = 3
    history_len: int = 1
    episode_len: int = 64
    gamma: float = 0.99

    # Training parameters
    batch_size: int = 128
    num_episodes: int = 131072  # batch_size * 1024
    use_episodic_freeze: bool = True
    use_baseline: bool = True
    learning_rate: float = 1e-3

    # Initial states
    init_states_tuple: Tuple[int, ...] = None

    # Experiment parameters
    seed: Optional[int] = None
    experiment_name: str = "congestion_game_experiment"
    run_id: Optional[str] = None

    # Nash equilibrium checking
    check_nash_equilibrium: bool = False  # New parameter


@dataclass
class SimulationResults:
    """Results from a single simulation run"""
    config: SimulationConfig
    agents_losses: List[List[float]]
    agents_returns: List[List[float]]
    episode_potentials: List[float]
    optimal_episode_discounted_potential: float
    argmax_episode_discounted_potential: float
    final_trajectory: List[Tuple[int, ...]]
    final_actions: List[torch.Tensor]
    training_time: float
    converged: bool = False
    final_grad_norms: Optional[List[List[float]]] = None

    # Nash equilibrium results
    is_nash_equilibrium: Optional[bool] = None  # New field
    nash_check_time: Optional[float] = None  # New field


class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles tuples and other non-serializable types"""

    def default(self, obj):
        if isinstance(obj, tuple):
            return {"__tuple__": list(obj)}
        if isinstance(obj, torch.Tensor):
            return {"__tensor__": obj.tolist()}
        if isinstance(obj, np.ndarray):
            return {"__array__": obj.tolist()}
        return super().default(obj)


def decode_json_tuples(obj):
    """Recursively convert lists back to tuples where appropriate"""
    if isinstance(obj, dict):
        if "__tuple__" in obj:
            return tuple(decode_json_tuples(item) for item in obj["__tuple__"])
        return {key: decode_json_tuples(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [decode_json_tuples(item) for item in obj]
    return obj


class SimulationManager:
    """Manages parallel simulation runs and result saving"""

    def __init__(self, base_output_dir: str = "simulation_results"):
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(exist_ok=True)

        # Setup logging
        self.setup_logging()

    def setup_logging(self):
        """Setup logging for the simulation manager"""
        log_file = self.base_output_dir / "simulation_manager.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def create_experiment_dir(self, experiment_name: str) -> Path:
        """Create a timestamped experiment directory"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = self.base_output_dir / f"{experiment_name}_{timestamp}"
        experiment_dir.mkdir(exist_ok=True)

        # Create subdirectories
        (experiment_dir / "configs").mkdir(exist_ok=True)
        (experiment_dir / "results").mkdir(exist_ok=True)
        (experiment_dir / "models").mkdir(exist_ok=True)
        (experiment_dir / "plots").mkdir(exist_ok=True)
        (experiment_dir / "logs").mkdir(exist_ok=True)

        return experiment_dir

    def save_config(self, config: SimulationConfig, experiment_dir: Path, run_id: str):
        """Save simulation configuration with proper tuple handling"""
        config_file = experiment_dir / "configs" / f"config_{run_id}.json"
        with open(config_file, 'w') as f:
            # Convert config to dict, handling non-serializable types
            config_dict = asdict(config)
            json.dump(config_dict, f, indent=2, cls=CustomJSONEncoder)

    def load_config(self, config_file: Path) -> SimulationConfig:
        """Load simulation configuration with proper tuple restoration"""
        with open(config_file, 'r') as f:
            config_dict = json.load(f)

        # Restore tuples
        config_dict = decode_json_tuples(config_dict)
        return SimulationConfig(**config_dict)

    def save_results(self, results: SimulationResults, experiment_dir: Path, run_id: str):
        """Save simulation results"""
        results_file = experiment_dir / "results" / f"results_{run_id}.pkl"
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)

        # Also save a summary as JSON for easy inspection
        summary = {
            "run_id": run_id,
            "training_time": results.training_time,
            "converged": results.converged,
            "optimal_potential": results.optimal_episode_discounted_potential,
            "final_potential": results.argmax_episode_discounted_potential,
            "final_trajectory": results.final_trajectory,
            "num_episodes": len(results.episode_potentials),
            "final_loss_means": [float(np.mean(losses[-10:])) for losses in results.agents_losses],
            "final_return_means": [float(np.mean(returns[-10:])) for returns in results.agents_returns],
            "is_nash_equilibrium": results.is_nash_equilibrium,
            "nash_check_time": results.nash_check_time
        }

        summary_file = experiment_dir / "results" / f"summary_{run_id}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, cls=CustomJSONEncoder)

    def save_models(self, agents: List[EpisodicAgent], experiment_dir: Path, run_id: str):
        """Save trained model parameters"""
        models_dir = experiment_dir / "models" / run_id
        models_dir.mkdir(exist_ok=True)

        for i, agent in enumerate(agents):
            model_file = models_dir / f"agent_{i}_policy.pth"
            torch.save(agent.policy_func.state_dict(), model_file)

    def check_nash_equilibrium(self, ecg: EpisodicCongestionGame) -> bool:
        """
        Check if the current policies form a Nash equilibrium
        This is adapted from the check_if_nash_eq method
        """
        ecg.reset()
        all_policy_maps = ecg.get_all_sampling_functions()
        agents_argmax_policy_maps = []
        for other_agent in ecg.agents:
            agents_argmax_policy_maps.append(other_agent.get_argmax_policy_map(ecg.H))
            other_agent.policy_map = agents_argmax_policy_maps[-1]

        # load precalculated argmax policies to boost up performance
        nash_filename = f'{ecg.N}_agents_{ecg.A}_states_actions_{ecg.H}_history_init_state_{tuple([agent.init_state.tolist()[0] for agent in ecg.agents])}_nash_bool_dict.pkl'
        if os.path.exists(nash_filename):
            with open(nash_filename, 'rb') as f:
                policies_nash_bool_dict = pickle.load(f)
        else:
            policies_nash_bool_dict = {}

        frozen = freeze_joint_policy(agents_argmax_policy_maps)
        if frozen in policies_nash_bool_dict:
            return policies_nash_bool_dict[frozen]

        # calc argmax return per agent
        argmax_returns = []
        _, _, episode_rewards, _ = ecg.do_episode(is_inference=False)
        for i in range(ecg.N):
            agent_rewards = torch.stack([step_reward[i] for step_reward in episode_rewards])
            returns = compute_discounted_returns(agent_rewards.detach(), gamma=ecg.gamma)
            argmax_returns.append(returns[0])

        for i, agent in enumerate(ecg.agents):
            all_agent_policy_maps = all_policy_maps[i]
            # agent i ran over all policy maps. find best discounted return for agent i. determine if best is current policy
            for ii, agent_i_policy in tqdm(enumerate(all_agent_policy_maps), desc=f"agent {i} policy comparison"):
                ecg.reset()
                # other agents frozen to argmax policy
                for j, other_agent in enumerate(ecg.agents):
                    if j != i:
                        other_agent.policy_map = agents_argmax_policy_maps[j]

                # agent i gets the fixed policy
                agent.policy_map = agent_i_policy

                _, _, episode_rewards, _ = ecg.do_episode(
                    is_inference=False)
                agent_rewards = torch.stack([step_reward[i] for step_reward in episode_rewards])
                returns = compute_discounted_returns(agent_rewards.detach(), gamma=ecg.gamma)

                # if found a policy that strictly beats the argmax
                if returns[0] > argmax_returns[i]:
                    policies_nash_bool_dict[frozen] = False
                    with open(nash_filename, 'wb') as f:
                        pickle.dump(policies_nash_bool_dict, f)
                    return False

        policies_nash_bool_dict[frozen] = True
        with open(nash_filename, 'wb') as f:
            pickle.dump(policies_nash_bool_dict, f)
        return True

    def run_single_simulation(self, config: SimulationConfig) -> Tuple[SimulationResults, List[EpisodicAgent]]:
        """Run a single simulation with the given configuration"""
        start_time = time.time()

        # Set seed if provided
        if config.seed is not None:
            torch.manual_seed(config.seed)
            np.random.seed(config.seed)

        # Create agents
        agents = []
        for i in range(config.num_agents):
            init_state = config.init_states_tuple[i]
            policy = DiscreteStatePolicyNoEmbeddings(
                state_vocab_sizes=(config.history_len + 1) * [config.state_dim],
                hidden_dim=config.action_dim,
                num_actions=config.action_dim
            )
            agent = EpisodicAgent(config.state_dim, config.action_dim,
                                  policy_func=policy, init_state=init_state)
            agents.append(agent)

        # Create environment
        ecg = EpisodicCongestionGame(
            agents=agents,
            num_actions=config.action_dim,
            g_func=g_func,
            u_funcs=[make_u_i(config.state_dim) for i in range(config.num_agents)],
            history_len=config.history_len,
            episode_len=config.episode_len,
            use_episodic_freeze=config.use_episodic_freeze
        )

        # Compute optimal policy for comparison
        optimal_policy = find_joint_optimum(
            num_agents=config.num_agents,
            num_states=config.state_dim,
            num_actions=config.action_dim,
            joint_reward_func=make_potential_func(config.state_dim),
            gamma=config.gamma
        )

        optimal_episode_discounted_potential, _ = evaluate_policy(
            ecg.agents, optimal_policy, make_potential_func(config.state_dim),
            gamma=config.gamma, episode_len=config.episode_len
        )

        # Train
        agents_losses, potentials, last_episode_potentials, last_actions, returns = train(
            ecg, config.num_episodes, batch_size=config.batch_size,
            use_baseline=config.use_baseline, learning_rate=config.learning_rate
        )

        # Get final policy performance
        ecg.reset()
        argmax_actions, max_logprobs, argmax_rewards, argmax_potentials = ecg.do_episode(is_inference=True)
        argmax_discounted_potentials = [(config.gamma ** t) * p for t, p in enumerate(argmax_potentials)]
        joint_actions_as_tuples = [tuple(action.tolist()) for action in argmax_actions]
        traj = get_joint_state_trajectory(config.init_states_tuple[:config.num_agents], joint_actions_as_tuples)

        training_time = time.time() - start_time

        # Check Nash equilibrium if requested
        is_nash_equilibrium = None
        nash_check_time = None
        if config.check_nash_equilibrium:
            nash_start_time = time.time()
            try:
                is_nash_equilibrium = self.check_nash_equilibrium(ecg)
                nash_check_time = time.time() - nash_start_time
            except Exception as e:
                self.logger.warning(f"Nash equilibrium check failed: {str(e)}")
                is_nash_equilibrium = None
                nash_check_time = None

        # Create results object
        results = SimulationResults(
            config=config,
            agents_losses=agents_losses,
            agents_returns=returns,
            episode_potentials=potentials,
            optimal_episode_discounted_potential=float(optimal_episode_discounted_potential),
            argmax_episode_discounted_potential=float(sum(argmax_discounted_potentials)),
            final_trajectory=traj,
            final_actions=argmax_actions,
            training_time=training_time,
            is_nash_equilibrium=is_nash_equilibrium,
            nash_check_time=nash_check_time
        )

        return results, agents

    def run_parallel_simulations(self, configs: List[SimulationConfig],
                                 max_workers: Optional[int] = None) -> Dict[str, SimulationResults]:
        """Run multiple simulations in parallel"""
        if max_workers is None:
            max_workers = min(len(configs), mp.cpu_count()-1)

        # Create experiment directory
        experiment_name = configs[0].experiment_name if configs else "parallel_experiment"
        experiment_dir = self.create_experiment_dir(experiment_name)

        self.logger.info(f"Starting {len(configs)} parallel simulations with {max_workers} workers")
        self.logger.info(f"Results will be saved to: {experiment_dir}")

        results = {}

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            future_to_config = {
                executor.submit(self.run_single_simulation, config): config
                for config in configs
            }

            # Process completed jobs
            for future in tqdm(as_completed(future_to_config), total=len(configs),
                               desc="Running simulations"):
                config = future_to_config[future]
                run_id = config.run_id or f"run_{len(results):04d}"

                try:
                    simulation_results, agents = future.result()

                    # Save results
                    self.save_config(config, experiment_dir, run_id)
                    self.save_results(simulation_results, experiment_dir, run_id)
                    self.save_models(agents, experiment_dir, run_id)

                    results[run_id] = simulation_results

                    nash_info = ""
                    if simulation_results.is_nash_equilibrium is not None:
                        nash_info = f" | Nash: {simulation_results.is_nash_equilibrium} ({simulation_results.nash_check_time:.2f}s)"

                    self.logger.info(
                        f"Completed simulation {run_id} in {simulation_results.training_time:.2f}s{nash_info}")

                except Exception as e:
                    self.logger.error(f"Simulation {run_id} failed: {str(e)}")

        # Save aggregated results
        self.save_aggregated_results(results, experiment_dir)

        self.logger.info(f"All simulations completed. Results saved to {experiment_dir}")
        return results

    def save_aggregated_results(self, results: Dict[str, SimulationResults], experiment_dir: Path):
        """Save aggregated analysis of all simulation results"""
        if not results:
            return

        # Compute statistics across runs
        final_potentials = [r.argmax_episode_discounted_potential for r in results.values()]
        optimal_potentials = [r.optimal_episode_discounted_potential for r in results.values()]
        training_times = [r.training_time for r in results.values()]

        # Nash equilibrium statistics
        nash_results = [r.is_nash_equilibrium for r in results.values() if r.is_nash_equilibrium is not None]
        nash_check_times = [r.nash_check_time for r in results.values() if r.nash_check_time is not None]

        aggregated = {
            "num_runs": len(results),
            "final_potential_stats": {
                "mean": float(np.mean(final_potentials)),
                "std": float(np.std(final_potentials)),
                "min": float(np.min(final_potentials)),
                "max": float(np.max(final_potentials))
            },
            "optimal_potential": float(np.mean(optimal_potentials)),  # Should be same for all
            "training_time_stats": {
                "mean": float(np.mean(training_times)),
                "std": float(np.std(training_times)),
                "min": float(np.min(training_times)),
                "max": float(np.max(training_times))
            },
            "convergence_rate": float(np.mean([r.converged for r in results.values()])),
            "run_ids": list(results.keys())
        }

        # Add Nash equilibrium statistics if available
        if nash_results:
            aggregated["nash_equilibrium_stats"] = {
                "num_checked": len(nash_results),
                "num_nash": int(sum(nash_results)),
                "nash_rate": float(np.mean(nash_results)),
                "nash_check_time_stats": {
                    "mean": float(np.mean(nash_check_times)) if nash_check_times else None,
                    "std": float(np.std(nash_check_times)) if nash_check_times else None,
                    "min": float(np.min(nash_check_times)) if nash_check_times else None,
                    "max": float(np.max(nash_check_times)) if nash_check_times else None
                }
            }

        aggregated_file = experiment_dir / "aggregated_results.json"
        with open(aggregated_file, 'w') as f:
            json.dump(aggregated, f, indent=2, cls=CustomJSONEncoder)


# Rest of the functions remain the same...
def compute_discounted_returns(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    """Computes the discounted return for a single trajectory."""
    returns = torch.zeros_like(rewards)
    R = torch.tensor(0)
    for t in reversed(range(rewards.shape[0])):
        R = rewards[t] + gamma * R
        returns[t] = R
    return returns


def find_joint_optimum(num_agents, num_states, num_actions, joint_reward_func, gamma=0.99, theta: float = 1e-1):
    """Find the joint optimal policy using value iteration"""
    joint_states = list(itertools.product(range(num_states), repeat=num_agents))
    joint_actions = list(itertools.product(range(num_actions), repeat=num_agents))

    V = torch.zeros([num_states] * num_agents)
    policy = {}

    while True:
        delta = 0
        V_new = V.clone()
        for s in joint_states:
            best_val = float('-inf')
            best_action = None
            for a in joint_actions:
                r = joint_reward_func(torch.tensor(s), torch.tensor(a))
                next_state = a
                val = r + gamma * V[next_state]
                if val > best_val:
                    best_val = val
                    best_action = a
            V_new[s] = best_val
            policy[s] = best_action
            delta = max(delta, abs(V_new[s] - V[s]))
        V = V_new
        if delta < num_agents * theta:
            break

    return policy


def train(env: EpisodicCongestionGame, num_episodes: int, batch_size: int = 1,
          use_baseline: bool = False, learning_rate: float = 1e-3):
    """Train the agents using REINFORCE"""
    independent_optimizers = []
    for agent in env.agents:
        independent_optimizers.append(optim.SGD(agent.policy_func.parameters(), lr=learning_rate))

    agents_losses = [[] for _ in env.agents]
    agents_returns = [[] for _ in env.agents]
    episode_potential_sums = []

    gamma = 0.99  # Should be passed as parameter

    for episode in tqdm(range(num_episodes // batch_size), desc="Training: "):
        all_agent_logprobs = [[] for _ in env.agents]
        all_agent_returns = [[] for _ in env.agents]
        all_episode_potentials = []

        for b in range(batch_size):
            env.reset()
            actions, logprobs, rewards, potentials = env.do_episode()

            for i, agent in enumerate(env.agents):
                agent_rewards = torch.stack([step_reward[i] for step_reward in rewards])
                returns = compute_discounted_returns(agent_rewards.detach(), gamma=gamma)

                if env.use_episodic_freeze:
                    agent_logprobs = torch.stack([logprob for action, logprob in agent.policy_map.values()])
                    all_agent_returns[i].append(returns[0])
                else:
                    agent_logprobs = torch.stack([step_logprobs[i] for step_logprobs in logprobs])
                    all_agent_returns[i].append(returns)

                all_agent_logprobs[i].append(agent_logprobs)

            potential_returns = compute_discounted_returns(torch.stack(potentials), gamma=gamma)
            all_episode_potentials.append(potential_returns[0])

        if use_baseline:
            agent_baselines = [torch.mean(torch.stack(returns), dim=0) for returns in all_agent_returns]

        agents_episode_losses = []
        agents_episode_returns = []

        for i, agent in enumerate(env.agents):
            loss = 0.0
            for logprobs, R in zip(all_agent_logprobs[i], all_agent_returns[i]):
                advantage = R - agent_baselines[i] if use_baseline else R
                if env.use_episodic_freeze:
                    loss += -torch.sum(logprobs) * advantage
                else:
                    loss += -torch.sum(logprobs * advantage)
            loss = loss / batch_size

            agents_episode_losses.append(loss)
            agents_episode_returns.append(torch.mean(torch.stack(all_agent_returns[i])).item())
            agents_losses[i].append(loss.item())
            agents_returns[i].append(agents_episode_returns[i])

        total_loss = sum(agents_episode_losses)
        for optimizer in independent_optimizers:
            optimizer.zero_grad()
        total_loss.backward()
        for optimizer in independent_optimizers:
            optimizer.step()

        episode_potential_sums.append(torch.mean(torch.stack(all_episode_potentials)).item())

    return agents_losses, episode_potential_sums, None, actions, agents_returns


def get_joint_state_trajectory(initial_state, joint_actions):
    """Reconstruct the joint state trajectory based on deterministic transitions."""
    trajectory = [initial_state]
    for action in joint_actions:
        trajectory.append(action)
    return trajectory


def make_random_init_tuple(n, max_s, rng=None):
    if rng is None:
        rng = random.Random()  # can pass a seedable Random instance
    # Pick n unique numbers from [0, max_s]
    init_states = rng.choices(range(max_s), k=n)
    return tuple(init_states)

def create_parameter_sweep_configs(base_config: SimulationConfig,
                                   param_grid: Dict[str, List[Any]]) -> List[SimulationConfig]:
    """Create configurations for parameter sweep"""
    configs = []

    # Get all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    for values in itertools.product(*param_values):
        config = SimulationConfig(**asdict(base_config))  # Copy base config
        # Update with current parameter values
        for param_name, value in zip(param_names, values):
            setattr(config, param_name, value)

        if config.init_states_tuple is None:
            rng = random.Random(config.seed)  # fixed seed for reproducibility
            config.init_states_tuple = make_random_init_tuple(config.num_agents, config.state_dim , rng)

        # Create unique run ID
        param_str = "_".join([f"{name}_{value}" for name, value in zip(param_names, values)])
        config.run_id = f"sweep_{param_str}"

        configs.append(config)

    return configs


if __name__ == '__main__':
    # Example usage
    manager = SimulationManager()
    
    # Define base configuration
    base_config = SimulationConfig(
        num_agents=3,
        state_dim=3,
        action_dim=3,
        num_episodes=32768,  # Smaller for example
        batch_size=128,
        experiment_name="congestion_game_parallel"
    )
    
    # Example 1: Run multiple seeds
    seed_configs = []
    for seed in range(5):
        config = SimulationConfig(**asdict(base_config))
        config.seed = seed
        config.run_id = f"seed_{seed}"
        seed_configs.append(config)
    
    print("Running multiple seeds...")
    seed_results = manager.run_parallel_simulations(seed_configs, max_workers=3)
    
    # Example 2: Parameter sweep
    param_grid = {
        'learning_rate': [1e-4, 1e-3, 1e-2],
        'batch_size': [64, 128, 256],
        'seed': [42, 123]
    }
    
    sweep_configs = create_parameter_sweep_configs(base_config, param_grid)
    print(f"\nRunning parameter sweep with {len(sweep_configs)} configurations...")
    sweep_results = manager.run_parallel_simulations(sweep_configs, max_workers=4)
