import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
from dataclasses import asdict
from collections import defaultdict

class ResultsAnalyzer:
    """Analyze and visualize results from parallel simulations, grouped by initial state"""
    
    def __init__(self, experiment_dir: str):
        self.experiment_dir = Path(experiment_dir)
        self.results = {}
        self.configs = {}
        self.results_by_state = defaultdict(dict)  # {initial_state: {run_id: result}}
        self.configs_by_state = defaultdict(dict)  # {initial_state: {run_id: config}}
        self.load_results()
        self.group_by_initial_state()
    
    def load_results(self):
        """Load all results and configurations from experiment directory"""
        results_dir = self.experiment_dir / "results"
        configs_dir = self.experiment_dir / "configs"
        
        if not results_dir.exists():
            raise ValueError(f"Results directory not found: {results_dir}")
        
        # Load results
        for results_file in results_dir.glob("results_*.pkl"):
            run_id = results_file.stem.replace("results_", "")
            with open(results_file, 'rb') as f:
                self.results[run_id] = pickle.load(f)
        
        # Load configs
        for config_file in configs_dir.glob("config_*.json"):
            run_id = config_file.stem.replace("config_", "")
            with open(config_file, 'r') as f:
                self.configs[run_id] = json.load(f)
            self.configs[run_id] = self.lists_to_tuples(self.configs[run_id])
        print(f"Loaded {len(self.results)} simulation results")

    def group_by_initial_state(self):
        """Group results and configs by initial state"""
        for run_id, result in self.results.items():
            # Get initial state from result or config
            initial_state = self._extract_initial_state(result, self.configs.get(run_id, {}))
            
            if initial_state is not None:
                self.results_by_state[initial_state][run_id] = result
                if run_id in self.configs:
                    self.configs_by_state[initial_state][run_id] = self.configs[run_id]
        
        print(f"Grouped results by {len(self.results_by_state)} different initial states")
        for state, runs in self.results_by_state.items():
            print(f"  Initial state {state}: {len(runs)} runs")

    def _extract_initial_state(self, result: Any, config: Dict) -> Optional[Tuple]:
        """Extract initial state from result or config"""
        # Try multiple ways to get initial state
        
        # 1. From result object
        if hasattr(result, 'initial_state'):
            return self._normalize_state(result.initial_state)
        
        # 2. From config
        if 'initial_state' in config:
            return self._normalize_state(config['initial_state'])
        
        # 3. From final trajectory (first state)
        if hasattr(result, 'final_trajectory') and result.final_trajectory:
            return self._normalize_state(result.final_trajectory[0])
        
        # 4. From episode states (first episode, first state)
        if hasattr(result, 'episode_states') and result.episode_states:
            first_episode = result.episode_states[0] if result.episode_states else None
            if first_episode:
                return self._normalize_state(first_episode[0])
        
        return None

    def _normalize_state(self, state: Any) -> Tuple:
        """Normalize state to tuple format for consistent hashing"""
        if isinstance(state, (list, tuple, np.ndarray)):
            # Convert to tuple, handling nested structures
            if isinstance(state, np.ndarray):
                state = state.tolist()
            return tuple(self._flatten_state(state))
        return tuple([state])

    def _flatten_state(self, state):
        """Recursively flatten state to handle nested structures"""
        if isinstance(state, (list, tuple)):
            result = []
            for item in state:
                if isinstance(item, (list, tuple, np.ndarray)):
                    result.extend(self._flatten_state(item))
                else:
                    result.append(float(item) if isinstance(item, (int, float, np.number)) else item)
            return result
        return [state]

    @staticmethod
    def lists_to_tuples(obj):
        if isinstance(obj, list):
            return tuple(ResultsAnalyzer.lists_to_tuples(item) for item in obj)
        elif isinstance(obj, dict):
            return {key: ResultsAnalyzer.lists_to_tuples(value) for key, value in obj.items()}
        else:
            return obj

    def plot_learning_curves_by_state(self, initial_state: Optional[Tuple] = None,
                                    run_ids: Optional[List[str]] = None, 
                                    metric: str = 'returns', save_path: Optional[str] = None):
        """Plot learning curves for specified runs within a specific initial state"""
        if initial_state is None:
            # Plot for the first available initial state
            initial_state = next(iter(self.results_by_state.keys()))
        
        state_results = self.results_by_state[initial_state]
        
        if run_ids is None:
            run_ids = list(state_results.keys())
        else:
            # Filter to only include runs that exist for this state
            run_ids = [rid for rid in run_ids if rid in state_results]
        
        if not run_ids:
            print(f"No runs found for initial state {initial_state}")
            return
        
        plt.figure(figsize=(15, 10))
        
        # Create subplots for each agent
        max_agents = max(len(state_results[rid].agents_returns) for rid in run_ids)
        
        for agent_idx in range(max_agents):
            plt.subplot(2, (max_agents + 1) // 2, agent_idx + 1)
            
            for run_id in run_ids:
                result = state_results[run_id]
                if metric == 'returns':
                    data = result.agents_returns[agent_idx]
                    ylabel = 'Returns'
                elif metric == 'losses':
                    data = result.agents_losses[agent_idx]
                    ylabel = 'Loss'
                else:
                    raise ValueError(f"Unknown metric: {metric}")
                
                # Apply smoothing
                smoothed_data = self.moving_average(data, window=20)
                plt.plot(smoothed_data, label=f'Run {run_id}', alpha=0.7)
            
            plt.xlabel('Episode')
            plt.ylabel(ylabel)
            plt.title(f'Agent {agent_idx} {metric.capitalize()}')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.suptitle(f'Learning Curves - Initial State: {initial_state}', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_potential_comparison_by_state(self, initial_state: Optional[Tuple] = None,
                                         run_ids: Optional[List[str]] = None,
                                         save_path: Optional[str] = None):
        """Plot potential function evolution for a specific initial state"""
        if initial_state is None:
            initial_state = next(iter(self.results_by_state.keys()))
        
        state_results = self.results_by_state[initial_state]
        
        if run_ids is None:
            run_ids = list(state_results.keys())
        else:
            run_ids = [rid for rid in run_ids if rid in state_results]
        
        if not run_ids:
            print(f"No runs found for initial state {initial_state}")
            return
        
        plt.figure(figsize=(12, 8))
        
        for run_id in run_ids:
            result = state_results[run_id]
            potentials = result.episode_potentials
            
            plt.plot(potentials, label=f'Run {run_id}', alpha=0.7)
            
            # Add optimal and final performance lines
            plt.axhline(y=result.optimal_episode_discounted_potential, 
                       color='red', linestyle='--', alpha=0.5, 
                       label='Joint Optimum' if run_id == run_ids[0] else "")
            plt.axhline(y=result.argmax_episode_discounted_potential,
                       color='green', linestyle=':', alpha=0.5,
                       label='Final Policy' if run_id == run_ids[0] else "")
        
        plt.xlabel('Episode')
        plt.ylabel('Discounted Potential')
        plt.title(f'Potential Function Evolution - Initial State: {initial_state}')
        # plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_convergence_analysis_by_state(self, initial_state: Optional[Tuple] = None,
                                         run_ids: Optional[List[str]] = None,
                                         save_path: Optional[str] = None):
        """Analyze convergence properties for a specific initial state"""
        if initial_state is None:
            initial_state = next(iter(self.results_by_state.keys()))
        
        state_results = self.results_by_state[initial_state]
        
        if run_ids is None:
            run_ids = list(state_results.keys())
        else:
            run_ids = [rid for rid in run_ids if rid in state_results]
        
        if not run_ids:
            print(f"No runs found for initial state {initial_state}")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Final performance distribution
        final_potentials = [state_results[rid].argmax_episode_discounted_potential for rid in run_ids]
        optimal_potentials = [state_results[rid].optimal_episode_discounted_potential for rid in run_ids]
        
        axes[0, 0].hist(final_potentials, bins=min(20, len(final_potentials)), alpha=0.7, label='Final Performance')
        axes[0, 0].axvline(np.mean(optimal_potentials), color='red', linestyle='--', label='Optimal')
        axes[0, 0].set_xlabel('Final Potential')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Final Performance Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Training time distribution
        training_times = [state_results[rid].training_time for rid in run_ids]
        axes[0, 1].hist(training_times, bins=min(20, len(training_times)), alpha=0.7)
        axes[0, 1].set_xlabel('Training Time (seconds)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Training Time Distribution')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Convergence gap vs training time
        convergence_gaps = [opt - final for opt, final in zip(optimal_potentials, final_potentials)]
        axes[1, 0].scatter(training_times, convergence_gaps, alpha=0.7)
        axes[1, 0].set_xlabel('Training Time (seconds)')
        axes[1, 0].set_ylabel('Convergence Gap (Optimal - Final)')
        axes[1, 0].set_title('Convergence Gap vs Training Time')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Parameter correlation
        state_configs = self.configs_by_state[initial_state]
        if len(set(state_configs[rid].get('learning_rate', 0) for rid in run_ids if rid in state_configs)) > 1:
            learning_rates = [state_configs[rid].get('learning_rate', 0) for rid in run_ids if rid in state_configs]
            corresponding_finals = [final_potentials[i] for i, rid in enumerate(run_ids) if rid in state_configs]
            axes[1, 1].scatter(learning_rates, corresponding_finals, alpha=0.7)
            axes[1, 1].set_xlabel('Learning Rate')
            axes[1, 1].set_ylabel('Final Potential')
            axes[1, 1].set_title('Learning Rate vs Performance')
            axes[1, 1].set_xscale('log')
        else:
            # Show final trajectory comparison instead
            for i, run_id in enumerate(run_ids[:5]):
                result = state_results[run_id]
                traj_length = len(result.final_trajectory) if hasattr(result, 'final_trajectory') else 0
                axes[1, 1].bar(i, traj_length, alpha=0.7, label=f'Run {run_id}')
            axes[1, 1].set_xlabel('Run')
            axes[1, 1].set_ylabel('Trajectory Length')
            axes[1, 1].set_title('Final Trajectory Lengths')
        
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'Convergence Analysis - Initial State: {initial_state}', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def create_summary_report_by_state(self, save_path: Optional[str] = None) -> str:
        """Generate a comprehensive summary report grouped by initial state"""
        report = []
        report.append("="*80)
        report.append("SIMULATION RESULTS SUMMARY REPORT (BY INITIAL STATE)")
        report.append("="*80)
        report.append(f"Experiment Directory: {self.experiment_dir}")
        report.append(f"Total Runs: {len(self.results)}")
        report.append(f"Number of Initial States: {len(self.results_by_state)}")
        report.append("")
        
        # Overall statistics
        all_final_potentials = []
        all_optimal_potentials = []
        all_training_times = []
        
        for initial_state, state_results in self.results_by_state.items():
            report.append("="*60)
            report.append(f"INITIAL STATE: {initial_state}")
            report.append("="*60)
            report.append(f"Number of runs for this state: {len(state_results)}")
            report.append("")
            
            # Performance Statistics for this state
            final_potentials = [r.argmax_episode_discounted_potential for r in state_results.values()]
            optimal_potentials = [r.optimal_episode_discounted_potential for r in state_results.values()]
            convergence_gaps = [opt - final for opt, final in zip(optimal_potentials, final_potentials)]
            training_times = [r.training_time for r in state_results.values()]
            
            # Add to overall stats
            all_final_potentials.extend(final_potentials)
            all_optimal_potentials.extend(optimal_potentials)
            all_training_times.extend(training_times)
            
            report.append("PERFORMANCE STATISTICS")
            report.append("-" * 40)
            report.append(f"Final Performance:")
            report.append(f"  Mean: {np.mean(final_potentials):.4f}")
            report.append(f"  Std:  {np.std(final_potentials):.4f}")
            report.append(f"  Min:  {np.min(final_potentials):.4f}")
            report.append(f"  Max:  {np.max(final_potentials):.4f}")
            report.append("")
            
            report.append(f"Optimal Performance: {np.mean(optimal_potentials):.4f}")
            report.append("")
            
            report.append(f"Convergence Gap (Optimal - Final):")
            report.append(f"  Mean: {np.mean(convergence_gaps):.4f}")
            report.append(f"  Std:  {np.std(convergence_gaps):.4f}")
            report.append(f"  Min:  {np.min(convergence_gaps):.4f}")
            report.append(f"  Max:  {np.max(convergence_gaps):.4f}")
            report.append("")
            
            # Training Statistics for this state
            report.append("TRAINING STATISTICS")
            report.append("-" * 40)
            report.append(f"Training Time (seconds):")
            report.append(f"  Mean: {np.mean(training_times):.2f}")
            report.append(f"  Std:  {np.std(training_times):.2f}")
            report.append(f"  Min:  {np.min(training_times):.2f}")
            report.append(f"  Max:  {np.max(training_times):.2f}")
            report.append("")
            
            # Best and Worst Runs for this state
            best_run_idx = np.argmax(final_potentials)
            worst_run_idx = np.argmin(final_potentials)
            run_ids = list(state_results.keys())
            best_run_id = run_ids[best_run_idx]
            worst_run_id = run_ids[worst_run_idx]
            
            state_configs = self.configs_by_state[initial_state]
            
            report.append("BEST AND WORST RUNS")
            report.append("-" * 40)
            report.append(f"Best Run: {best_run_id}")
            report.append(f"  Final Potential: {final_potentials[best_run_idx]:.4f}")
            report.append(f"  Training Time: {training_times[best_run_idx]:.2f}s")
            report.append(f"  Config: {state_configs.get(best_run_id, 'N/A')}")
            report.append("")
            
            report.append(f"Worst Run: {worst_run_id}")
            report.append(f"  Final Potential: {final_potentials[worst_run_idx]:.4f}")
            report.append(f"  Training Time: {training_times[worst_run_idx]:.2f}s")
            report.append(f"  Config: {state_configs.get(worst_run_id, 'N/A')}")
            report.append("")
            
            # Nash equilibrium stats for this state
            nash_stats = self.get_nash_equilibrium_stats_for_state(initial_state)
            if nash_stats['num_checked'] > 0:
                report.append("NASH EQUILIBRIUM STATISTICS")
                report.append("-" * 40)
                report.append(f"Runs checked: {nash_stats['num_checked']}")
                report.append(f"Nash equilibria found: {nash_stats['num_nash']}")
                report.append(f"Nash equilibrium rate: {nash_stats['nash_rate']:.2%}")
                report.append("")
        
        # Overall summary across all initial states
        report.append("="*80)
        report.append("OVERALL SUMMARY (ALL INITIAL STATES)")
        report.append("="*80)
        
        report.append("OVERALL PERFORMANCE STATISTICS")
        report.append("-" * 40)
        report.append(f"Final Performance (all states):")
        report.append(f"  Mean: {np.mean(all_final_potentials):.4f}")
        report.append(f"  Std:  {np.std(all_final_potentials):.4f}")
        report.append(f"  Min:  {np.min(all_final_potentials):.4f}")
        report.append(f"  Max:  {np.max(all_final_potentials):.4f}")
        report.append("")
        
        all_convergence_gaps = [opt - final for opt, final in zip(all_optimal_potentials, all_final_potentials)]
        report.append(f"Convergence Gap (all states):")
        report.append(f"  Mean: {np.mean(all_convergence_gaps):.4f}")
        report.append(f"  Std:  {np.std(all_convergence_gaps):.4f}")
        report.append(f"  Min:  {np.min(all_convergence_gaps):.4f}")
        report.append(f"  Max:  {np.max(all_convergence_gaps):.4f}")
        report.append("")
        
        report.append("OVERALL TRAINING STATISTICS")
        report.append("-" * 40)
        report.append(f"Training Time (seconds, all states):")
        report.append(f"  Mean: {np.mean(all_training_times):.2f}")
        report.append(f"  Std:  {np.std(all_training_times):.2f}")
        report.append(f"  Min:  {np.min(all_training_times):.2f}")
        report.append(f"  Max:  {np.max(all_training_times):.2f}")
        report.append("")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
        
        return report_text

    def get_nash_equilibrium_stats_for_state(self, initial_state: Tuple) -> Dict:
        """Get Nash equilibrium statistics for a specific initial state"""
        state_results = self.results_by_state[initial_state]
        nash_results = []
        nash_check_times = []

        for result in state_results.values():
            if hasattr(result, 'is_nash_equilibrium') and result.is_nash_equilibrium is not None:
                nash_results.append(result.is_nash_equilibrium)
                if hasattr(result, 'nash_check_time') and result.nash_check_time is not None:
                    nash_check_times.append(result.nash_check_time)

        if not nash_results:
            return {"num_checked": 0, "message": "No Nash equilibrium checks performed"}

        return {
            "num_checked": len(nash_results),
            "num_nash": sum(nash_results),
            "nash_rate": np.mean(nash_results),
        }

    def get_nash_equilibrium_stats(self) -> Dict:
        """Get Nash equilibrium statistics from loaded results (overall)"""
        overall_stats = {"by_state": {}, "overall": {"num_checked": 0, "num_nash": 0}}
        
        total_checked = 0
        total_nash = 0
        
        for initial_state in self.results_by_state.keys():
            state_stats = self.get_nash_equilibrium_stats_for_state(initial_state)
            overall_stats["by_state"][str(initial_state)] = state_stats
            
            if state_stats["num_checked"] > 0:
                total_checked += state_stats["num_checked"]
                total_nash += state_stats["num_nash"]
        
        if total_checked > 0:
            overall_stats["overall"] = {
                "num_checked": total_checked,
                "num_nash": total_nash,
                "nash_rate": total_nash / total_checked
            }
        else:
            overall_stats["overall"] = {"num_checked": 0, "message": "No Nash equilibrium checks performed"}
        
        return overall_stats

    def export_results_to_csv_by_state(self, save_path: str):
        """Export results to CSV with initial state information"""
        data = []
        
        for initial_state, state_results in self.results_by_state.items():
            state_configs = self.configs_by_state[initial_state]
            
            for run_id, result in state_results.items():
                config = state_configs.get(run_id, {})
                
                row = {
                    'run_id': run_id,
                    'initial_state': str(initial_state),
                    'final_potential': result.argmax_episode_discounted_potential,
                    'optimal_potential': result.optimal_episode_discounted_potential,
                    'convergence_gap': result.optimal_episode_discounted_potential - result.argmax_episode_discounted_potential,
                    'training_time': result.training_time,
                    'num_episodes': len(result.episode_potentials),
                    'final_trajectory_length': len(result.final_trajectory) if hasattr(result, 'final_trajectory') else 0,
                }
                
                # Add Nash equilibrium info if available
                if hasattr(result, 'is_nash_equilibrium') and result.is_nash_equilibrium is not None:
                    row['is_nash_equilibrium'] = result.is_nash_equilibrium
                
                # Add config parameters
                for key, value in config.items():
                    if key not in row:
                        row[f'config_{key}'] = value
                
                # Add final losses and returns for each agent
                for i, losses in enumerate(result.agents_losses):
                    row[f'agent_{i}_final_loss'] = losses[-1] if losses else None
                    row[f'agent_{i}_final_return'] = result.agents_returns[i][-1] if result.agents_returns[i] else None
                
                data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(save_path, index=False)
        print(f"Results exported to {save_path}")

    def compare_runs_by_state(self, initial_state: Optional[Tuple] = None, 
                            run_ids: Optional[List[str]] = None, 
                            save_dir: Optional[str] = None):
        """Generate comprehensive comparison plots for specific runs within an initial state"""
        if initial_state is None:
            initial_state = next(iter(self.results_by_state.keys()))
        
        if save_dir:
            save_dir = Path(save_dir) / f"state_{hash(initial_state) % 10000}"
            save_dir.mkdir(parents=True, exist_ok=True)
        
        state_results = self.results_by_state[initial_state]
        if run_ids is None:
            run_ids = list(state_results.keys())
        else:
            run_ids = [rid for rid in run_ids if rid in state_results]
        
        if not run_ids:
            print(f"No runs found for initial state {initial_state}")
            return
        
        # Learning curves
        self.plot_learning_curves_by_state(initial_state, run_ids, metric='returns',
                                         save_path=save_dir / 'learning_curves_returns.png' if save_dir else None)
        self.plot_learning_curves_by_state(initial_state, run_ids, metric='losses',
                                         save_path=save_dir / 'learning_curves_losses.png' if save_dir else None)
        
        # Potential comparison
        self.plot_potential_comparison_by_state(initial_state, run_ids,
                                              save_path=save_dir / 'potential_comparison.png' if save_dir else None)
        
        # Convergence analysis
        self.plot_convergence_analysis_by_state(initial_state, run_ids,
                                               save_path=save_dir / 'convergence_analysis.png' if save_dir else None)
        
        # Nash equilibrium stats for this state
        nash_stats = self.get_nash_equilibrium_stats_for_state(initial_state)
        print(f"Nash equilibrium stats for state {initial_state}: {nash_stats}")
        
        print(f"Comparison plots for initial state {initial_state} {'saved to ' + str(save_dir) if save_dir else 'displayed'}")

    def compare_across_initial_states(self, metric: str = 'final_potential', save_path: Optional[str] = None):
        """Compare performance across different initial states"""
        state_stats = {}
        
        for initial_state, state_results in self.results_by_state.items():
            if metric == 'final_potential':
                values = [r.argmax_episode_discounted_potential for r in state_results.values()]
            elif metric == 'convergence_gap':
                values = [r.optimal_episode_discounted_potential - r.argmax_episode_discounted_potential 
                         for r in state_results.values()]
            elif metric == 'training_time':
                values = [r.training_time for r in state_results.values()]
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            state_stats[str(initial_state)] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'count': len(values)
            }
        
        # Create comparison plot
        plt.figure(figsize=(12, 8))
        
        states = list(state_stats.keys())
        means = [state_stats[s]['mean'] for s in states]
        stds = [state_stats[s]['std'] for s in states]
        
        plt.errorbar(range(len(states)), means, yerr=stds, fmt='o', capsize=5)
        plt.xlabel('Initial State')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(f'{metric.replace("_", " ").title()} Comparison Across Initial States')
        plt.xticks(range(len(states)), [f'State {i}' for i in range(len(states))], rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return state_stats

    @staticmethod
    def moving_average(data: List[float], window: int = 10) -> List[float]:
        """Apply moving average smoothing"""
        if len(data) < window:
            return data
        return [np.mean(data[i:i+window]) for i in range(len(data) - window + 1)]

    def list_initial_states(self):
        """List all unique initial states and their run counts"""
        print("Available Initial States:")
        print("-" * 50)
        for i, (state, runs) in enumerate(self.results_by_state.items()):
            print(f"{i}: {state} ({len(runs)} runs)")
        return list(self.results_by_state.keys())

    def get_state_by_index(self, index: int) -> Tuple:
        """Get initial state by index for easier access"""
        states = list(self.results_by_state.keys())
        if 0 <= index < len(states):
            return states[index]
        else:
            raise IndexError(f"Index {index} out of range. Available indices: 0-{len(states)-1}")

    def plot_all_states_summary(self, save_dir: Optional[str] = None):
        """Create summary plots comparing all initial states"""
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True)
        
        # 1. Performance comparison across states
        self.compare_across_initial_states('final_potential', 
                                         save_path=save_dir / 'states_final_potential.png' if save_dir else None)
        
        # 2. Convergence gap comparison
        self.compare_across_initial_states('convergence_gap',
                                         save_path=save_dir / 'states_convergence_gap.png' if save_dir else None)
        
        # 3. Training time comparison
        self.compare_across_initial_states('training_time',
                                         save_path=save_dir / 'states_training_time.png' if save_dir else None)
        
        # 4. Nash equilibrium rates by state (if available)
        nash_stats = self.get_nash_equilibrium_stats()
        if nash_stats['overall']['num_checked'] > 0:
            plt.figure(figsize=(10, 6))
            states_with_nash = []
            nash_rates = []
            
            for state_str, stats in nash_stats['by_state'].items():
                if stats['num_checked'] > 0:
                    states_with_nash.append(state_str)
                    nash_rates.append(stats['nash_rate'])
            
            if states_with_nash:
                plt.bar(range(len(states_with_nash)), nash_rates)
                plt.xlabel('Initial State')
                plt.ylabel('Nash Equilibrium Rate')
                plt.title('Nash Equilibrium Rate by Initial State')
                plt.xticks(range(len(states_with_nash)), 
                          [f'State {i}' for i in range(len(states_with_nash))], rotation=45)
                plt.ylim(0, 1)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                if save_dir:
                    plt.savefig(save_dir / 'states_nash_rates.png', dpi=300, bbox_inches='tight')
                plt.show()


# Example usage with the new state-based functionality
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        experiment_dir = sys.argv[1]
    else:
        # Use most recent experiment directory
        base_dir = Path("simulation_results")
        if base_dir.exists():
            experiment_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
            if experiment_dirs:
                experiment_dir = max(experiment_dirs, key=lambda x: x.stat().st_mtime)
            else:
                print("No experiment directories found!")
                exit(1)
        else:
            print("No simulation_results directory found!")
            exit(1)
    
    print(f"Analyzing results from: {experiment_dir}")
    
    # Create analyzer
    analyzer = ResultsAnalyzer(experiment_dir)
    
    # List available initial states
    print("\n" + "="*80)
    states = analyzer.list_initial_states()
    
    # Generate summary report by state
    report = analyzer.create_summary_report_by_state()
    print("\n" + report)
    
    # Save report
    report_path = Path(experiment_dir) / "analysis_report_by_state.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nDetailed report saved to: {report_path}")
    
    # Export to CSV with state information
    csv_path = Path(experiment_dir) / "results_summary_by_state.csv"
    analyzer.export_results_to_csv_by_state(csv_path)
    
    # Generate comparison plots for all states
    comparison_dir = Path(experiment_dir) / "comparison_plots_by_state"
    analyzer.plot_all_states_summary(save_dir=comparison_dir)
    
    # Generate detailed analysis for each initial state (limit to first few states to avoid too many plots)
    max_states_to_plot = min(3, len(states))  # Plot detailed analysis for up to 3 states
    
    for i in range(max_states_to_plot):
        if i == 0:
            state = (0.0, 0.0, 1.0)
        else:
            state = analyzer.get_state_by_index(i)
        print(f"\nGenerating detailed plots for initial state {i}: {state}")
        
        # Get some sample runs for this state (up to 10 runs)
        state_run_ids = list(analyzer.results_by_state[state].keys())
        sample_runs = state_run_ids
        
        analyzer.compare_runs_by_state(state, sample_runs, save_dir=comparison_dir)
    
    # Overall Nash equilibrium statistics
    nash_stats = analyzer.get_nash_equilibrium_stats()
    print(f"\nOverall Nash equilibrium stats: {nash_stats}")
    
    print(f"\nAll analysis complete! Results saved to: {experiment_dir}")
    print(f"Comparison plots saved to: {comparison_dir}")