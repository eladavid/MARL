import pickle
import os
from typing import Dict, Any

from simulations.sim_utils import MultiAgentSimulation


def analyze_sim_global_is_not_nash(sim_data: Dict[str, Any]):
    simulation = sim_data['simulation']
    optimal_value_func = simulation.multi_agent.calc_value_function(simulation.multi_agent.optimal_policy)
    nash_value_funcs = sim_data['nash_policies_joint_value_funcs']
    diffs = [optimal_value_func - v for v in nash_value_funcs]

    return 0

if __name__ == "__main__":
    global_is_not_nash_dir = os.path.join(os.getcwd(), "..\sim_res\\regular")
    sim_pkl9 = os.path.join(global_is_not_nash_dir, 'sim_0_regular.pkl')
    with open(sim_pkl9, 'rb') as f:
        sim_data = pickle.load(f)
    analyze_sim_global_is_not_nash(sim_data)
    # for sim_pkl in os.listdir(global_is_not_nash_dir):
    #     sim_fullpath = os.path.join(global_is_not_nash_dir, sim_pkl)
    #     with open(sim_fullpath, 'rb') as f:
    #         sim = pickle.load(f)
    #     analyze_sim_global_is_not_nash(sim)