# Enhanced Multi-Agent RL Simulation System

This enhanced version of your multi-agent RL simulation adds **parallel execution** and **comprehensive result saving** capabilities. Here's how to use the new features:

## 🚀 Quick Start

### 1. Run a Predefined Experiment

```bash
# Run a quick test (fast, for debugging)
python experiment_configs.py quick_test

# Run baseline experiment with multiple seeds
python experiment_configs.py baseline

# Run learning rate sweep
python experiment_configs.py learning_rate_sweep
```

### 2. Analyze Results

```bash
# Analyze the most recent experiment
python results_analyzer.py

# Analyze a specific experiment directory
python results_analyzer.py simulation_results/baseline_experiment_20240805_143022
```

## 📁 Directory Structure

After running experiments, you'll get organized output:

```
simulation_results/
├── experiment_name_timestamp/
│   ├── configs/           # Configuration files for each run
│   │   ├── config_run_001.json
│   │   └── ...
│   ├── results/           # Detailed results and summaries
│   │   ├── results_run_001.pkl
│   │   ├── summary_run_001.json
│   │   └── ...
│   ├── models/            # Trained model parameters
│   │   ├── run_001/
│   │   │   ├── agent_0_policy.pth
│   │   │   └── ...
│   │   └── ...
│   ├── plots/             # Generated visualizations
│   ├── logs/              # Training logs
│   ├── aggregated_results.json
│   ├── analysis_report.txt
│   └── results_summary.csv
```

## 🔧 Programming Interface

### Basic Usage

```python
from simulation_runner import SimulationManager, SimulationConfig

# Create simulation manager
manager = SimulationManager()

# Define a configuration
config = SimulationConfig(
    num_agents=3,
    state_dim=3,
    num_episodes=32768,
    learning_rate=1e-3,
    seed=42,
    experiment_name="my_experiment"
)

# Run single simulation
results, agents = manager.run_single_simulation(config)

# Run multiple simulations in parallel
configs = [config1, config2, config3, ...]  # List of configs
results_dict = manager.run_parallel_simulations(configs, max_workers=4)
```

### Parameter Sweeps

```python
from experiment_configs import create_parameter_sweep_configs

# Define base configuration
base_config = SimulationConfig(
    num_agents=3,
    state_dim=3,
    experiment_name="lr_sweep"
)

# Define parameter grid
param_grid = {
    'learning_rate': [1e-4, 1e-3, 1e-2],
    'batch_size': [64, 128, 256],
    'seed': [42, 123, 456]
}

# Generate all combinations
configs = create_parameter_sweep_configs(base_config, param_grid)
print(f"Generated {len(configs)} configurations")  # 3 × 3 × 3 = 27

# Run the sweep
manager = SimulationManager()
results = manager.run_parallel_simulations(configs)
```

## 📊 Available Experiments

| Experiment | Description | Configurations |
|------------|-------------|----------------|
| `quick_test` | Fast test for debugging | 2 seeds, short episodes |
|