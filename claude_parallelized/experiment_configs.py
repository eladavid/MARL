#!/usr/bin/env python3
"""
Experiment Configuration Generator for Multi-Agent RL Simulations

This script provides pre-defined experiment configurations and utilities
for generating parameter sweeps and multi-seed runs.
"""

import argparse
import json
from pathlib import Path
from dataclasses import asdict
from typing import Dict, List, Any
import itertools

# Import your simulation modules (adjust paths as needed)
from parallel_simulation import SimulationConfig, SimulationManager, create_parameter_sweep_configs


def create_baseline_experiment() -> List[SimulationConfig]:
    """Create a baseline experiment with multiple seeds"""
    base_config = SimulationConfig(
        num_agents=3,
        state_dim=3,
        action_dim=3,
        history_len=1,
        episode_len=64,
        gamma=0.99,
        batch_size=128,
        num_episodes=32768,  # 256 batches
        use_episodic_freeze=True,
        use_baseline=True,
        learning_rate=1e-3,
        init_states_tuple=(1, 1, 0, 2, 0),
        experiment_name="baseline_experiment"
    )
    
    configs = []
    for seed in range(10):  # 10 different seeds
        config = SimulationConfig(**asdict(base_config))
        config.seed = seed
        config.run_id = f"baseline_seed_{seed}"
        configs.append(config)
    
    return configs


def create_learning_rate_sweep() -> List[SimulationConfig]:
    """Create a learning rate sweep experiment"""
    base_config = SimulationConfig(
        num_agents=3,
        state_dim=3,
        action_dim=3,
        history_len=1,
        episode_len=64,
        gamma=0.99,
        batch_size=128,
        num_episodes=32768,
        use_episodic_freeze=True,
        use_baseline=True,
        init_states_tuple=(1, 1, 0, 2, 0),
        experiment_name="learning_rate_sweep"
    )
    
    param_grid = {
        'learning_rate': [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
        'seed': [42, 123, 456]  # Multiple seeds per learning rate
    }
    
    return create_parameter_sweep_configs(base_config, param_grid)


def create_architecture_comparison() -> List[SimulationConfig]:
    """Compare different batch sizes and episode lengths"""
    base_config = SimulationConfig(
        num_agents=3,
        state_dim=3,
        action_dim=3,
        history_len=1,
        gamma=0.99,
        use_episodic_freeze=True,
        use_baseline=True,
        learning_rate=1e-3,
        init_states_tuple=(1, 1, 0, 2, 0),
        experiment_name="architecture_comparison"
    )
    
    param_grid = {
        'batch_size': [64, 128, 256],
        'episode_len': [32, 64, 128],
        'seed': [42, 123]
    }
    
    configs = create_parameter_sweep_configs(base_config, param_grid)
    
    # Adjust num_episodes to maintain similar total training steps
    for config in configs:
        config.num_episodes = max(16384, config.batch_size * 128)  # At least 128 batches
    
    return configs


def create_freeze_comparison() -> List[SimulationConfig]:
    """Compare episodic freeze vs standard REINFORCE"""
    base_config = SimulationConfig(
        num_agents=3,
        state_dim=3,
        action_dim=3,
        history_len=1,
        episode_len=64,
        gamma=0.99,
        batch_size=128,
        num_episodes=32768,
        use_baseline=True,
        learning_rate=1e-3,
        init_states_tuple=(1, 1, 0, 2, 0),
        experiment_name="freeze_comparison"
    )
    
    param_grid = {
        'use_episodic_freeze': [True, False],
        'use_baseline': [True, False],
        'seed': range(5)
    }
    
    return create_parameter_sweep_configs(base_config, param_grid)


def create_scalability_test() -> List[SimulationConfig]:
    """Test scalability with different numbers of agents"""
    configs = []
    
    for num_agents in [2, 3, 4, 5]:
        base_config = SimulationConfig(
            num_agents=num_agents,
            state_dim=3,
            action_dim=3,
            history_len=1,
            episode_len=64,
            gamma=0.99,
            batch_size=64,  # Smaller batches for larger systems
            num_episodes=16384,  # Fewer episodes for larger systems
            use_episodic_freeze=True,
            use_baseline=True,
            learning_rate=1e-3,
            init_states_tuple=tuple(range(num_agents)),  # Simple initial states
            experiment_name=f"scalability_test_{num_agents}_agents"
        )
        
        # Run multiple seeds for each agent count
        for seed in range(3):
            config = SimulationConfig(**asdict(base_config))
            config.seed = seed
            config.run_id = f"agents_{num_agents}_seed_{seed}"
            configs.append(config)
    
    return configs


def create_quick_test() -> List[SimulationConfig]:
    """Create a quick test configuration for debugging"""
    base_config = SimulationConfig(
        num_agents=3,
        state_dim=3,
        action_dim=3,
        history_len=1,
        episode_len=32,  # Shorter episodes
        gamma=0.99,
        batch_size=32,   # Smaller batches
        num_episodes=64,  # Much fewer episodes
        use_episodic_freeze=True,
        use_baseline=True,
        learning_rate=1e-3,
        init_states_tuple=(1, 1, 0),
        experiment_name="quick_test"
    )
    
    configs = []
    for seed in [42, 123]:  # Just 2 seeds
        config = SimulationConfig(**asdict(base_config))
        config.seed = seed
        config.run_id = f"quick_test_seed_{seed}"
        configs.append(config)
    
    return configs


def save_configs_to_file(configs: List[SimulationConfig], filename: str):
    """Save configurations to a JSON file for later use"""
    config_dicts = [asdict(config) for config in configs]
    with open(filename, 'w') as f:
        json.dump(config_dicts, f, indent=2)
    print(f"Saved {len(configs)} configurations to {filename}")


def load_configs_from_file(filename: str) -> List[SimulationConfig]:
    """Load configurations from a JSON file"""
    with open(filename, 'r') as f:
        config_dicts = json.load(f)
    
    configs = []
    for config_dict in config_dicts:
        # Convert tuple fields back from lists
        if 'init_states_tuple' in config_dict and isinstance(config_dict['init_states_tuple'], list):
            config_dict['init_states_tuple'] = tuple(config_dict['init_states_tuple'])
        configs.append(SimulationConfig(**config_dict))
    
    return configs


def run_experiment(experiment_name: str, max_workers: int = None, dry_run: bool = False):
    """Run a predefined experiment"""
    
    experiment_functions = {
        'baseline': create_baseline_experiment,
        'learning_rate_sweep': create_learning_rate_sweep,
        'architecture_comparison': create_architecture_comparison,
        'freeze_comparison': create_freeze_comparison,
        'scalability_test': create_scalability_test,
        'quick_test': create_quick_test
    }
    
    if experiment_name not in experiment_functions:
        print(f"Unknown experiment: {experiment_name}")
        print(f"Available experiments: {list(experiment_functions.keys())}")
        return
    
    print(f"Setting up {experiment_name} experiment...")
    configs = experiment_functions[experiment_name]()
    
    print(f"Generated {len(configs)} simulation configurations")
    
    if dry_run:
        print("Dry run - not executing simulations")
        print("First few configurations:")
        for i, config in enumerate(configs[:3]):
            print(f"  Config {i+1}: {config.run_id}")
            print(f"    - Learning rate: {config.learning_rate}")
            print(f"    - Batch size: {config.batch_size}")
            print(f"    - Num episodes: {config.num_episodes}")
            print(f"    - Seed: {config.seed}")
        return configs
    
    # Run the simulations
    manager = SimulationManager()
    results = manager.run_parallel_simulations(configs, max_workers=max_workers)
    
    print(f"Experiment {experiment_name} completed!")
    print(f"Results available for {len(results)} runs")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run multi-agent RL experiments")
    parser.add_argument('experiment', choices=[
        'baseline', 'learning_rate_sweep', 'architecture_comparison',
        'freeze_comparison', 'scalability_test', 'quick_test'
    ], help='Experiment to run')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers (default: auto)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show configurations without running')
    parser.add_argument('--save-configs', type=str,
                       help='Save configurations to JSON file instead of running')
    
    args = parser.parse_args()
    
    if args.save_configs:
        # Generate and save configurations
        experiment_functions = {
            'baseline': create_baseline_experiment,
            'learning_rate_sweep': create_learning_rate_sweep,
            'architecture_comparison': create_architecture_comparison,
            'freeze_comparison': create_freeze_comparison,
            'scalability_test': create_scalability_test,
            'quick_test': create_quick_test
        }
        
        configs = experiment_functions[args.experiment]()
        save_configs_to_file(configs, args.save_configs)
    else:
        # Run experiment
        run_experiment(args.experiment, max_workers=args.workers, dry_run=args.dry_run)


# Additional utility functions for custom experiments

def create_custom_parameter_sweep(base_params: Dict[str, Any], 
                                param_grid: Dict[str, List[Any]], 
                                experiment_name: str = "custom_sweep") -> List[SimulationConfig]:
    """Create a custom parameter sweep experiment
    
    Args:
        base_params: Base configuration parameters
        param_grid: Dictionary of parameters and their values to sweep
        experiment_name: Name for the experiment
    
    Returns:
        List of simulation configurations
    """
    base_config = SimulationConfig(**base_params, experiment_name=experiment_name)
    return create_parameter_sweep_configs(base_config, param_grid)


def create_multi_seed_experiment(base_params: Dict[str, Any], 
                                seeds: List[int],
                                experiment_name: str = "multi_seed") -> List[SimulationConfig]:
    """Create an experiment with multiple random seeds
    
    Args:
        base_params: Base configuration parameters
        seeds: List of random seeds to use
        experiment_name: Name for the experiment
    
    Returns:
        List of simulation configurations
    """
    configs = []
    base_config = SimulationConfig(**base_params, experiment_name=experiment_name)
    
    for seed in seeds:
        config = SimulationConfig(**asdict(base_config))
        config.seed = seed
        config.run_id = f"{experiment_name}_seed_{seed}"
        configs.append(config)
    
    return configs


# Example of how to create a completely custom experiment
def create_my_custom_experiment():
    """Example of creating a custom experiment"""
    
    # Define base parameters
    base_params = {
        'num_agents': 4,
        'state_dim': 5,
        'action_dim': 5,
        'history_len': 2,
        'episode_len': 128,
        'gamma': 0.95,
        'batch_size': 256,
        'num_episodes': 65536,
        'use_episodic_freeze': True,
        'use_baseline': True,
        'init_states_tuple': (0, 1, 2, 3),
        'experiment_name': 'my_custom_experiment'
    }
    
    # Define parameter sweep
    param_grid = {
        'learning_rate': [1e-4, 1e-3, 1e-2],
        'gamma': [0.9, 0.95, 0.99],
        'seed': [42, 123, 456, 789, 999]
    }
    
    return create_custom_parameter_sweep(base_params, param_grid)


if __name__ == '__main__':
    # If called directly, run the main function
    main()
