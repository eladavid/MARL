#!/usr/bin/env python3
"""
Nash Equilibrium Checker for Multi-Agent RL Simulations

This module handles Nash equilibrium checking outside of the training process,
clustering simulations by initial state tuples and running checks in parallel.
"""

import os
import pickle
import time
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
import torch
import numpy as np
from tqdm import tqdm
import json

# These imports should match your actual module structure
# Adjust the import paths as needed for your specific setup
try:
    from congestion_game.episodic_agent import EpisodicAgent
    from congestion_game.episodic_congestion_game import EpisodicCongestionGame
    from congestion_game.policies import DiscreteStatePolicyNoEmbeddings
    from congestion_game.reward_functions import g_func, make_u_i
    from congestion_game.utils import compute_discounted_returns, freeze_joint_policy
except ImportError as e:
    print(f"Import error: {e}")
    print("Please adjust the import paths to match your module structure")
    raise


@dataclass
class NashCheckConfig:
    """Configuration for Nash equilibrium checking"""
    num_agents: int
    state_dim: int
    action_dim: int
    history_len: int
    episode_len: int
    gamma: float
    init_states_tuple: Tuple[int, ...]


@dataclass
class NashCheckResult:
    """Result from Nash equilibrium check"""
    run_id: str
    init_states_tuple: Tuple[int, ...]
    is_nash_equilibrium: bool
    check_time: float
    error_message: Optional[str] = None


class NashEquilibriumChecker:
    """Handles Nash equilibrium checking for trained policies"""
    
    def __init__(self, base_output_dir: str = "simulation_results"):
        self.base_output_dir = Path(base_output_dir)
    
    def load_trained_model(self, model_dir: Path, config: NashCheckConfig) -> List[EpisodicAgent]:
        """Load trained agents from saved model parameters"""
        agents = []
        
        for i in range(config.num_agents):
            # Create agent with same configuration as training
            init_state = config.init_states_tuple[i]
            policy = DiscreteStatePolicyNoEmbeddings(
                state_vocab_sizes=(config.history_len + 1) * [config.state_dim],
                hidden_dim=config.action_dim,
                num_actions=config.action_dim
            )
            agent = EpisodicAgent(
                config.state_dim, 
                config.action_dim,
                policy_func=policy, 
                init_state=init_state
            )
            
            # Load trained parameters
            model_file = model_dir / f"agent_{i}_policy.pth"
            if model_file.exists():
                agent.policy_func.load_state_dict(torch.load(model_file, map_location='cpu'))
            else:
                raise FileNotFoundError(f"Model file not found: {model_file}")
            
            agents.append(agent)
        
        return agents
    
    def check_nash_equilibrium_for_policy(self, agents: List[EpisodicAgent], 
                                        config: NashCheckConfig) -> bool:
        """Check if the given policy configuration is a Nash equilibrium"""
        # Create environment
        ecg = EpisodicCongestionGame(
            agents=agents,
            num_actions=config.action_dim,
            g_func=g_func,
            u_funcs=[make_u_i(config.state_dim) for _ in range(config.num_agents)],
            history_len=config.history_len,
            episode_len=config.episode_len,
            use_episodic_freeze=True  # Use episodic freeze for Nash check
        )
        
        ecg.reset()
        agents_argmax_policy_maps = []
        
        # Get argmax policies for all agents
        for agent in ecg.agents:
            agents_argmax_policy_maps.append(agent.get_argmax_policy_map(ecg.H))
            agent.policy_map = agents_argmax_policy_maps[-1]
        
        # Check if we've already computed this Nash equilibrium
        nash_filename = f'{ecg.N}_agents_{ecg.A}_states_actions_{ecg.H}_history_init_state_{tuple([agent.init_state.tolist()[0] for agent in ecg.agents])}_nash_bool_dict.pkl'
        
        if os.path.exists(nash_filename):
            with open(nash_filename, 'rb') as f:
                policies_nash_bool_dict = pickle.load(f)
        else:
            policies_nash_bool_dict = {}
        
        frozen = freeze_joint_policy(agents_argmax_policy_maps)
        if frozen in policies_nash_bool_dict:
            return policies_nash_bool_dict[frozen]

        # Calculate argmax return per agent
        argmax_returns = []
        _, _, episode_rewards, _ = ecg.do_episode(is_inference=False)
        for i in range(ecg.N):
            agent_rewards = torch.stack([step_reward[i] for step_reward in episode_rewards])
            returns = compute_discounted_returns(agent_rewards.detach(), gamma=config.gamma)
            argmax_returns.append(returns[0])
        
        # Check if any agent can improve by deviating
        all_policy_maps = ecg.get_all_sampling_functions()
        for i, agent in enumerate(ecg.agents):
            all_agent_policy_maps = all_policy_maps[i]
            
            for agent_i_policy in tqdm(all_agent_policy_maps):
                ecg.reset()
                
                # Fix other agents to argmax policy
                for j, other_agent in enumerate(ecg.agents):
                    if j != i:
                        other_agent.policy_map = agents_argmax_policy_maps[j]
                
                # Agent i gets the alternative policy
                agent.policy_map = agent_i_policy
                
                _, _, episode_rewards, _ = ecg.do_episode(is_inference=False)
                agent_rewards = torch.stack([step_reward[i] for step_reward in episode_rewards])
                returns = compute_discounted_returns(agent_rewards.detach(), gamma=config.gamma)
                
                # If found a policy that strictly beats the argmax
                if returns[0] > argmax_returns[i]:
                    policies_nash_bool_dict[frozen] = False
                    with open(nash_filename, 'wb') as f:
                        pickle.dump(policies_nash_bool_dict, f)
                    return False
        
        # No profitable deviations found - it's a Nash equilibrium
        policies_nash_bool_dict[frozen] = True
        with open(nash_filename, 'wb') as f:
            pickle.dump(policies_nash_bool_dict, f)
        return True


def process_single_cluster(args) -> List[NashCheckResult]:
    """Process a single cluster of runs (for parallel execution across clusters)"""
    experiment_dir, cluster_init_states, run_ids, nash_configs_dict = args
    
    # Create a fresh checker instance for this process
    cluster_results = []
    
    print(f"Processing cluster {cluster_init_states} with {len(run_ids)} runs")
    
    # Sequential execution within cluster due to shared Nash pickle file
    for run_id in run_ids:
        model_dir = Path(experiment_dir) / "models" / run_id
        if not model_dir.exists():
            print(f"Warning: Model directory not found for run {run_id}, skipping")
            continue
        
        # Reconstruct NashCheckConfig from dict
        nash_config = NashCheckConfig(**nash_configs_dict[run_id])
        args = (run_id, model_dir, nash_config)
        
        try:
            result = check_single_run_nash(args)
            cluster_results.append(result)
        except Exception as e:
            print(f"Nash check failed for run {run_id}: {str(e)}")
            # Create error result
            error_result = NashCheckResult(
                run_id=run_id,
                init_states_tuple=cluster_init_states,
                is_nash_equilibrium=False,
                check_time=0.0,
                error_message=str(e)
            )
            cluster_results.append(error_result)
    
    return cluster_results


def check_single_run_nash(args) -> NashCheckResult:
    """Check Nash equilibrium for a single run (for parallel execution)"""
    run_id, model_dir, config = args
    
    start_time = time.time()
    checker = NashEquilibriumChecker()
    
    try:
        # Load trained agents
        agents = checker.load_trained_model(model_dir, config)
        
        # Check Nash equilibrium
        is_nash = checker.check_nash_equilibrium_for_policy(agents, config)
        
        check_time = time.time() - start_time
        
        return NashCheckResult(
            run_id=run_id,
            init_states_tuple=config.init_states_tuple,
            is_nash_equilibrium=is_nash,
            check_time=check_time
        )
    
    except Exception as e:
        check_time = time.time() - start_time
        return NashCheckResult(
            run_id=run_id,
            init_states_tuple=config.init_states_tuple,
            is_nash_equilibrium=False,
            check_time=check_time,
            error_message=str(e)
        )


class ExperimentNashChecker:
    """Main class for checking Nash equilibria across experiment results"""
    
    def __init__(self, experiment_dir: str):
        self.experiment_dir = Path(experiment_dir)
        self.clusters = defaultdict(list)  # init_states_tuple -> list of run_ids
        self.configs = {}  # run_id -> config
        self.results = {}  # run_id -> original results
        self.nash_results = {}  # run_id -> NashCheckResult
    
    def load_experiment_data(self):
        """Load experiment configurations and results"""
        print("Loading experiment data...")
        
        # Load configurations
        configs_dir = self.experiment_dir / "configs"
        if not configs_dir.exists():
            raise ValueError(f"Configs directory not found: {configs_dir}")
        
        for config_file in configs_dir.glob("config_*.json"):
            run_id = config_file.stem.replace("config_", "")
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            # Convert lists back to tuples where needed
            if 'init_states_tuple' in config_data and isinstance(config_data['init_states_tuple'], list):
                config_data['init_states_tuple'] = tuple(config_data['init_states_tuple'])
            
            self.configs[run_id] = config_data
        
        # Load existing results
        results_dir = self.experiment_dir / "results"
        if results_dir.exists():
            for results_file in results_dir.glob("results_*.pkl"):
                run_id = results_file.stem.replace("results_", "")
                with open(results_file, 'rb') as f:
                    self.results[run_id] = pickle.load(f)
        
        print(f"Loaded {len(self.configs)} configurations and {len(self.results)} results")
    
    def cluster_by_initial_states(self):
        """Cluster simulations by their initial state tuples"""
        print("Clustering simulations by initial state tuples...")
        
        for run_id, config in self.configs.items():
            init_states = config.get('init_states_tuple')
            if init_states:
                self.clusters[init_states].append(run_id)
        
        print(f"Created {len(self.clusters)} clusters:")
        for init_states, run_ids in self.clusters.items():
            print(f"  {init_states}: {len(run_ids)} runs")
    
    def create_nash_check_configs(self) -> Dict[str, NashCheckConfig]:
        """Create NashCheckConfig objects for each run"""
        nash_configs = {}
        
        for run_id, config in self.configs.items():
            nash_config = NashCheckConfig(
                num_agents=config['num_agents'],
                state_dim=config['state_dim'],
                action_dim=config['action_dim'],
                history_len=config['history_len'],
                episode_len=config['episode_len'],
                gamma=config['gamma'],
                init_states_tuple=config['init_states_tuple']
            )
            nash_configs[run_id] = nash_config
        
        return nash_configs
    
    def check_cluster_nash_equilibria(self, cluster_init_states: Tuple[int, ...], 
                                    run_ids: List[str], nash_configs: Dict[str, NashCheckConfig]) -> List[NashCheckResult]:
        """Check Nash equilibria for all runs in a cluster (sequential within cluster due to shared file)"""
        print(f"Checking Nash equilibria for cluster {cluster_init_states} ({len(run_ids)} runs)")
        
        cluster_results = []
        
        # Sequential execution within cluster due to shared Nash pickle file
        for run_id in tqdm(run_ids, desc=f"Nash checks for {cluster_init_states}"):
            model_dir = self.experiment_dir / "models" / run_id
            if not model_dir.exists():
                print(f"Warning: Model directory not found for run {run_id}, skipping")
                continue
            
            nash_config = nash_configs[run_id]
            args = (run_id, model_dir, nash_config)
            
            try:
                result = check_single_run_nash(args)
                cluster_results.append(result)
            except Exception as e:
                print(f"Nash check failed for run {run_id}: {str(e)}")
                # Create error result
                error_result = NashCheckResult(
                    run_id=run_id,
                    init_states_tuple=cluster_init_states,
                    is_nash_equilibrium=False,
                    check_time=0.0,
                    error_message=str(e)
                )
                cluster_results.append(error_result)
        
        return cluster_results
    
    def run_all_nash_checks(self, max_workers: Optional[int] = None, debug: bool = False) -> Dict[str, NashCheckResult]:
        """Run Nash equilibrium checks for all clusters (parallel across clusters)"""
        if max_workers is None:
            max_workers = min(len(self.clusters), mp.cpu_count() - 1)
        
        print(f"Starting Nash equilibrium checks for {len(self.clusters)} clusters using {max_workers} workers...")
        
        nash_configs = self.create_nash_check_configs()
        all_nash_results = {}
        
        if max_workers == 1 or len(self.clusters) == 1 or debug==True:
            # Sequential execution across clusters
            for cluster_init_states, run_ids in self.clusters.items():
                cluster_results = self.check_cluster_nash_equilibria(
                    cluster_init_states, run_ids, nash_configs
                )
                
                # Store results
                for result in cluster_results:
                    all_nash_results[result.run_id] = result
                    self.nash_results[result.run_id] = result
        else:
            # Parallel execution across clusters
            # Prepare arguments for each cluster
            cluster_args = []
            for cluster_init_states, run_ids in self.clusters.items():
                # Convert NashCheckConfig objects to dicts for serialization
                cluster_nash_configs = {
                    run_id: asdict(nash_configs[run_id]) 
                    for run_id in run_ids if run_id in nash_configs
                }
                cluster_args.append((
                    str(self.experiment_dir),
                    cluster_init_states,
                    run_ids,
                    cluster_nash_configs
                ))
            
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit cluster processing jobs
                future_to_cluster = {
                    executor.submit(process_single_cluster, args): args[1]  # args[1] is cluster_init_states
                    for args in cluster_args
                }
                
                # Process completed cluster jobs
                for future in tqdm(as_completed(future_to_cluster), total=len(self.clusters),
                                 desc="Processing clusters"):
                    cluster_init_states = future_to_cluster[future]
                    try:
                        cluster_results = future.result()
                        
                        # Store results
                        for result in cluster_results:
                            all_nash_results[result.run_id] = result
                            self.nash_results[result.run_id] = result
                        
                        print(f"Completed cluster {cluster_init_states}: {len(cluster_results)} checks")
                        
                    except Exception as e:
                        print(f"Cluster {cluster_init_states} processing failed: {str(e)}")
        
        return all_nash_results
    
    def _process_cluster_worker(self, cluster_init_states: Tuple[int, ...], 
                              run_ids: List[str], nash_configs: Dict[str, NashCheckConfig]) -> List[NashCheckResult]:
        """Worker function for processing a single cluster (for parallel execution across clusters)"""
        # This function will be executed in a separate process
        # We need to recreate the checker instance for this process
        checker = ExperimentNashChecker(self.experiment_dir)
        return checker.check_cluster_nash_equilibria(cluster_init_states, run_ids, nash_configs)
    
    def update_original_results(self):
        """Update original results with Nash equilibrium information"""
        print("Updating original results with Nash equilibrium data...")
        
        updated_count = 0
        for run_id, nash_result in self.nash_results.items():
            if run_id in self.results:
                # Update the original result object
                original_result = self.results[run_id]
                original_result.is_nash_equilibrium = nash_result.is_nash_equilibrium
                original_result.nash_check_time = nash_result.check_time
                
                # Save updated result
                results_file = self.experiment_dir / "results" / f"results_{run_id}.pkl"
                with open(results_file, 'wb') as f:
                    pickle.dump(original_result, f)
                
                # Update summary JSON as well
                summary_file = self.experiment_dir / "results" / f"summary_{run_id}.json"
                if summary_file.exists():
                    with open(summary_file, 'r') as f:
                        summary = json.load(f)
                    
                    summary['is_nash_equilibrium'] = nash_result.is_nash_equilibrium
                    summary['nash_check_time'] = nash_result.check_time
                    if nash_result.error_message:
                        summary['nash_check_error'] = nash_result.error_message
                    
                    with open(summary_file, 'w') as f:
                        json.dump(summary, f, indent=2)
                
                updated_count += 1
        
        print(f"Updated {updated_count} result files with Nash equilibrium data")
    
    def display_nash_stats(self):
        """Display comprehensive Nash equilibrium statistics"""
        if not self.nash_results:
            print("No Nash equilibrium results available")
            return
        
        print("\n" + "="*80)
        print("NASH EQUILIBRIUM ANALYSIS")
        print("="*80)
        
        # Overall statistics
        total_checked = len(self.nash_results)
        successful_checks = sum(1 for r in self.nash_results.values() if r.error_message is None)
        nash_equilibria = sum(1 for r in self.nash_results.values() 
                             if r.error_message is None and r.is_nash_equilibrium)
        failed_checks = total_checked - successful_checks
        
        print(f"Total runs checked: {total_checked}")
        print(f"Successful checks: {successful_checks}")
        print(f"Failed checks: {failed_checks}")
        print(f"Nash equilibria found: {nash_equilibria}")
        print(f"Nash equilibrium rate: {nash_equilibria/successful_checks*100:.1f}%" if successful_checks > 0 else "N/A")
        
        # Timing statistics
        check_times = [r.check_time for r in self.nash_results.values() if r.error_message is None]
        if check_times:
            print(f"\nNash check timing:")
            print(f"  Mean time: {np.mean(check_times):.2f}s")
            print(f"  Std time: {np.std(check_times):.2f}s")
            print(f"  Min time: {np.min(check_times):.2f}s")
            print(f"  Max time: {np.max(check_times):.2f}s")
            print(f"  Total time: {np.sum(check_times):.2f}s")
        
        # Per-cluster statistics
        print(f"\nPer-cluster statistics:")
        for init_states, run_ids in self.clusters.items():
            cluster_results = [self.nash_results[run_id] for run_id in run_ids 
                             if run_id in self.nash_results]
            
            if not cluster_results:
                continue
            
            cluster_successful = sum(1 for r in cluster_results if r.error_message is None)
            cluster_nash = sum(1 for r in cluster_results 
                             if r.error_message is None and r.is_nash_equilibrium)
            
            nash_rate = f"{cluster_nash/cluster_successful*100:.1f}%" if cluster_successful > 0 else "N/A"
            print(f"  {init_states}: {cluster_nash}/{cluster_successful} Nash ({nash_rate})")
        
        # Failed checks details
        if failed_checks > 0:
            print(f"\nFailed checks:")
            for run_id, result in self.nash_results.items():
                if result.error_message:
                    print(f"  {run_id}: {result.error_message}")
    
    def save_nash_summary(self):
        """Save Nash equilibrium summary to file"""
        if not self.nash_results:
            return
        
        # Prepare summary data
        summary = {
            "total_checked": len(self.nash_results),
            "successful_checks": sum(1 for r in self.nash_results.values() if r.error_message is None),
            "nash_equilibria_found": sum(1 for r in self.nash_results.values() 
                                       if r.error_message is None and r.is_nash_equilibrium),
            "failed_checks": sum(1 for r in self.nash_results.values() if r.error_message is not None),
            "nash_results": {}
        }
        
        # Add individual results
        for run_id, result in self.nash_results.items():
            summary["nash_results"][run_id] = {
                "is_nash_equilibrium": result.is_nash_equilibrium,
                "check_time": result.check_time,
                "init_states_tuple": result.init_states_tuple,
                "error_message": result.error_message
            }
        
        # Add cluster-wise statistics
        summary["cluster_stats"] = {}
        for init_states, run_ids in self.clusters.items():
            cluster_results = [self.nash_results[run_id] for run_id in run_ids 
                             if run_id in self.nash_results]
            
            if cluster_results:
                cluster_successful = sum(1 for r in cluster_results if r.error_message is None)
                cluster_nash = sum(1 for r in cluster_results 
                                 if r.error_message is None and r.is_nash_equilibrium)
                
                summary["cluster_stats"][str(init_states)] = {
                    "total_runs": len(cluster_results),
                    "successful_checks": cluster_successful,
                    "nash_equilibria": cluster_nash,
                    "nash_rate": cluster_nash / cluster_successful if cluster_successful > 0 else None
                }
        
        # Save to file
        nash_summary_file = self.experiment_dir / "nash_equilibrium_summary.json"
        with open(nash_summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)  # default=str handles tuples
        
        print(f"Nash equilibrium summary saved to: {nash_summary_file}")


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Check Nash equilibria for experiment results")
    parser.add_argument('experiment_dir', help='Path to experiment directory')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers across clusters (default: auto)')
    parser.add_argument('--update-results', action='store_true',
                       help='Update original result files with Nash equilibrium data')
    
    args = parser.parse_args()
    
    # Create and run checker
    checker = ExperimentNashChecker(args.experiment_dir)
    checker.load_experiment_data()
    checker.cluster_by_initial_states()
    
    # Run Nash checks
    nash_results = checker.run_all_nash_checks(max_workers=args.workers)
    
    # Display and save results
    checker.display_nash_stats()
    checker.save_nash_summary()
    
    # Update original results if requested
    if args.update_results:
        checker.update_original_results()
    
    print(f"\nNash equilibrium checking completed for {len(nash_results)} runs")


if __name__ == '__main__':
    main()
