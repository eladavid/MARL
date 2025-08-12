import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from dataclasses import asdict

class ResultsAnalyzer:
    """Analyze and visualize results from parallel simulations"""
    
    def __init__(self, experiment_dir: str):
        self.experiment_dir = Path(experiment_dir)
        self.results = {}
        self.configs = {}
        self.load_results()
    
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

    @staticmethod
    def lists_to_tuples(obj):
        if isinstance(obj, list):
            return tuple(ResultsAnalyzer.lists_to_tuples(item) for item in obj)
        elif isinstance(obj, dict):
            return {key: ResultsAnalyzer.lists_to_tuples(value) for key, value in obj.items()}
        else:
            return obj

    def plot_learning_curves(self, run_ids: Optional[List[str]] = None, 
                           metric: str = 'returns', save_path: Optional[str] = None):
        """Plot learning curves for specified runs"""
        if run_ids is None:
            run_ids = list(self.results.keys())
        
        plt.figure(figsize=(15, 10))
        
        # Create subplots for each agent
        max_agents = max(len(self.results[rid].agents_returns) for rid in run_ids)
        
        for agent_idx in range(max_agents):
            plt.subplot(2, (max_agents + 1) // 2, agent_idx + 1)
            
            for run_id in run_ids:
                result = self.results[run_id]
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
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_potential_comparison(self, run_ids: Optional[List[str]] = None,
                                save_path: Optional[str] = None):
        """Plot potential function evolution and compare to optimal"""
        if run_ids is None:
            run_ids = list(self.results.keys())
        
        plt.figure(figsize=(12, 8))
        
        for run_id in run_ids:
            result = self.results[run_id]
            potentials = result.episode_potentials
            
            # Apply smoothing
            # smoothed_potentials = self.moving_average(potentials, window=50)
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
        plt.title('Potential Function Evolution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_parameter_sweep_heatmap(self, param1: str, param2: str, 
                                   metric: str = 'final_potential',
                                   save_path: Optional[str] = None):
        """Create heatmap for parameter sweep results"""
        # Extract data for heatmap
        data_points = []
        for run_id, result in self.results.items():
            config = self.configs[run_id]
            
            if metric == 'final_potential':
                value = result.argmax_episode_discounted_potential
            elif metric == 'training_time':
                value = result.training_time
            elif metric == 'convergence_gap':
                value = (result.optimal_episode_discounted_potential - 
                        result.argmax_episode_discounted_potential)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            data_points.append({
                param1: config[param1],
                param2: config[param2],
                'value': value,
                'run_id': run_id
            })
        
        # Convert to DataFrame and pivot
        df = pd.DataFrame(data_points)
        heatmap_data = df.pivot_table(values='value', index=param1, columns=param2, aggfunc='mean')
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='viridis')
        plt.title(f'{metric.replace("_", " ").title()} - {param1} vs {param2}')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_convergence_analysis(self, run_ids: Optional[List[str]] = None,
                                save_path: Optional[str] = None):
        """Analyze convergence properties across runs"""
        if run_ids is None:
            run_ids = list(self.results.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Final performance distribution
        final_potentials = [self.results[rid].argmax_episode_discounted_potential for rid in run_ids]
        optimal_potentials = [self.results[rid].optimal_episode_discounted_potential for rid in run_ids]
        
        axes[0, 0].hist(final_potentials, bins=20, alpha=0.7, label='Final Performance')
        axes[0, 0].axvline(np.mean(optimal_potentials), color='red', linestyle='--', label='Optimal')
        axes[0, 0].set_xlabel('Final Potential')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Final Performance Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Training time distribution
        training_times = [self.results[rid].training_time for rid in run_ids]
        axes[0, 1].hist(training_times, bins=20, alpha=0.7)
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
        
        # 4. Parameter correlation (if sweep data available)
        if len(set(self.configs[rid].get('learning_rate', 0) for rid in run_ids)) > 1:
            learning_rates = [self.configs[rid].get('learning_rate', 0) for rid in run_ids]
            axes[1, 1].scatter(learning_rates, final_potentials, alpha=0.7)
            axes[1, 1].set_xlabel('Learning Rate')
            axes[1, 1].set_ylabel('Final Potential')
            axes[1, 1].set_title('Learning Rate vs Performance')
            axes[1, 1].set_xscale('log')
        else:
            # Show final trajectory comparison instead
            for i, run_id in enumerate(run_ids[:5]):  # Show first 5 runs
                result = self.results[run_id]
                traj_lengths = [len(result.final_trajectory)]
                axes[1, 1].bar(i, traj_lengths[0], alpha=0.7, label=f'Run {run_id}')
            axes[1, 1].set_xlabel('Run')
            axes[1, 1].set_ylabel('Trajectory Length')
            axes[1, 1].set_title('Final Trajectory Lengths')
        
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_summary_report(self, save_path: Optional[str] = None) -> str:
        """Generate a comprehensive summary report"""
        report = []
        report.append("="*80)
        report.append("SIMULATION RESULTS SUMMARY REPORT")
        report.append("="*80)
        report.append(f"Experiment Directory: {self.experiment_dir}")
        report.append(f"Total Runs: {len(self.results)}")
        report.append("")
        
        # Performance Statistics
        final_potentials = [r.argmax_episode_discounted_potential for r in self.results.values()]
        optimal_potentials = [r.optimal_episode_discounted_potential for r in self.results.values()]
        convergence_gaps = [opt - final for opt, final in zip(optimal_potentials, final_potentials)]
        training_times = [r.training_time for r in self.results.values()]
        
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
        
        # Training Statistics
        report.append("TRAINING STATISTICS")
        report.append("-" * 40)
        report.append(f"Training Time (seconds):")
        report.append(f"  Mean: {np.mean(training_times):.2f}")
        report.append(f"  Std:  {np.std(training_times):.2f}")
        report.append(f"  Min:  {np.min(training_times):.2f}")
        report.append(f"  Max:  {np.max(training_times):.2f}")
        report.append("")
        
        # Best and Worst Runs
        best_run_idx = np.argmax(final_potentials)
        worst_run_idx = np.argmin(final_potentials)
        best_run_id = list(self.results.keys())[best_run_idx]
        worst_run_id = list(self.results.keys())[worst_run_idx]
        
        report.append("BEST AND WORST RUNS")
        report.append("-" * 40)
        report.append(f"Best Run: {best_run_id}")
        report.append(f"  Final Potential: {final_potentials[best_run_idx]:.4f}")
        report.append(f"  Training Time: {training_times[best_run_idx]:.2f}s")
        report.append(f"  Config: {self.configs.get(best_run_id, 'N/A')}")
        report.append("")
        
        report.append(f"Worst Run: {worst_run_id}")
        report.append(f"  Final Potential: {final_potentials[worst_run_idx]:.4f}")
        report.append(f"  Training Time: {training_times[worst_run_idx]:.2f}s")
        report.append(f"  Config: {self.configs.get(worst_run_id, 'N/A')}")
        report.append("")
        
        # Parameter Analysis (if available)
        if self.configs:
            report.append("PARAMETER ANALYSIS")
            report.append("-" * 40)
            
            # Find parameters that vary across runs
            all_params = set()
            for config in self.configs.values():
                all_params.update(config.keys())
            
            varying_params = {}
            for param in all_params:
                values = [config.get(param) for config in self.configs.values()]
                unique_values = list(set(v for v in values if v is not None))
                if len(unique_values) > 1:
                    varying_params[param] = unique_values
            
            if varying_params:
                report.append("Parameters that vary across runs:")
                for param, values in varying_params.items():
                    report.append(f"  {param}: {values}")
            else:
                report.append("All runs used identical parameters")
            report.append("")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
        
        return report_text
    
    def export_results_to_csv(self, save_path: str):
        """Export results to CSV for further analysis"""
        data = []
        
        for run_id, result in self.results.items():
            config = self.configs.get(run_id, {})
            
            row = {
                'run_id': run_id,
                'final_potential': result.argmax_episode_discounted_potential,
                'optimal_potential': result.optimal_episode_discounted_potential,
                'convergence_gap': result.optimal_episode_discounted_potential - result.argmax_episode_discounted_potential,
                'training_time': result.training_time,
                'num_episodes': len(result.episode_potentials),
                'final_trajectory_length': len(result.final_trajectory),
            }
            
            # Add config parameters
            for key, value in config.items():
                if key not in row:  # Avoid overwriting existing columns
                    row[f'config_{key}'] = value
            
            # Add final losses and returns for each agent
            for i, losses in enumerate(result.agents_losses):
                row[f'agent_{i}_final_loss'] = losses[-1] if losses else None
                row[f'agent_{i}_final_return'] = result.agents_returns[i][-1] if result.agents_returns[i] else None
            
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(save_path, index=False)
        print(f"Results exported to {save_path}")
    
    def plot_trajectory_comparison(self, run_ids: Optional[List[str]] = None,
                                 save_path: Optional[str] = None):
        """Compare final trajectories across runs"""
        if run_ids is None:
            run_ids = list(self.results.keys())[:5]  # Show first 5 runs
        
        fig, axes = plt.subplots(1, len(run_ids), figsize=(4*len(run_ids), 4))
        if len(run_ids) == 1:
            axes = [axes]
        
        for i, run_id in enumerate(run_ids):
            result = self.results[run_id]
            trajectory = result.final_trajectory
            
            # Create a simple trajectory plot
            if len(trajectory) > 0 and len(trajectory[0]) >= 2:
                # Plot trajectory for first 3 agents
                agent_0_states = [state[0] for state in trajectory]
                agent_1_states = [state[1] for state in trajectory]
                agent_2_states = [state[2] for state in trajectory]
                
                axes[i].plot(agent_0_states, label='Agent 0', marker='o')
                axes[i].plot(agent_1_states, label='Agent 1', marker='s')
                axes[i].plot(agent_2_states, label='Agent 2', marker='d')
                axes[i].set_xlabel('Time Step')
                axes[i].set_ylabel('State')
                axes[i].set_title(f'Run {run_id}\nFinal Potential: {result.argmax_episode_discounted_potential:.3f}')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def moving_average(data: List[float], window: int = 10) -> List[float]:
        """Apply moving average smoothing"""
        if len(data) < window:
            return data
        return [np.mean(data[i:i+window]) for i in range(len(data) - window + 1)]

    def get_nash_equilibrium_stats(self) -> Dict:
        """Get Nash equilibrium statistics from loaded results"""
        nash_results = []
        nash_check_times = []

        for result in self.results.values():
            if hasattr(result, 'is_nash_equilibrium') and result.is_nash_equilibrium is not None:
                nash_results.append(result.is_nash_equilibrium)
                if not result.is_nash_equilibrium:
                    print(f"result is not nash!. run name: {result.config.run_id}. final trajectory: {result.final_trajectory}")
                if hasattr(result, 'nash_check_time') and result.nash_check_time is not None:
                    nash_check_times.append(result.nash_check_time)

        if not nash_results:
            return {"num_checked": 0, "message": "No Nash equilibrium checks performed"}

        return {
            "num_checked": len(nash_results),
            "num_nash": sum(nash_results),
            "nash_rate": np.mean(nash_results),
        }


    def compare_runs(self, run_ids: List[str], save_dir: Optional[str] = None):
        """Generate comprehensive comparison plots for specific runs"""
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True)
        
        # Learning curves
        self.plot_learning_curves(run_ids, metric='returns', 
                                save_path=save_dir / 'learning_curves_returns.png' if save_dir else None)
        self.plot_learning_curves(run_ids, metric='losses',
                                save_path=save_dir / 'learning_curves_losses.png' if save_dir else None)
        
        # Potential comparison
        self.plot_potential_comparison(run_ids,
                                     save_path=save_dir / 'potential_comparison.png' if save_dir else None)
        
        # Trajectory comparison
        self.plot_trajectory_comparison(run_ids,
                                      save_path=save_dir / 'trajectory_comparison.png' if save_dir else None)

        nash_stats = self.get_nash_equilibrium_stats()
        print()
        print(f"Nash equilibrium stats: {nash_stats}")
        
        print(f"Comparison plots {'saved to ' + str(save_dir) if save_dir else 'displayed'}")


# Example usage
if __name__ == '__main__':
    # Example of how to use the analyzer
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
    
    # Generate summary report
    report = analyzer.create_summary_report()
    print(report)
    
    # Save report
    report_path = Path(experiment_dir) / "analysis_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")
    
    # Export to CSV
    csv_path = Path(experiment_dir) / "results_summary.csv"
    analyzer.export_results_to_csv(csv_path)
    
    # Generate comparison plots for all runs
    comparison_dir = Path(experiment_dir) / "comparison_plots"
    run_ids = list(analyzer.results.keys())
    ids_to_sample = [829, 913, 1036, 0, 2,65, 6, 14, 144, 332, 1049, 553, 992]
    analyzer.compare_runs([run_ids[i] for i  in ids_to_sample], save_dir=comparison_dir)  # Compare first 10 runs