from typing import List, Dict, Set, Sequence

import numpy as np
import os
import matplotlib
import pickle

from RL_utils import normalize_value_function
from reward_functions import Reward, SimpleTwoAgentsReward, SeperableMultiAgentReward
from simulations.sim_utils import MDP, Agent, MultiAgent, MultiAgentSimulation

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

import random


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    # Example usage:

    # Define MDP parameters
    states = {0, 1}
    actions = {0, 1}
    transition_prob1 = {
        0: {0: [1., 0.], 1: [0., 1.]},
        1: {0: [1., 0.], 1: [0., 1.]}
    }

    transition_prob2 = {
        0: {0: [0., 1.], 1: [0., 1.]},
        1: {0: [1., 0.], 1: [1., 0.]}
    }

    start_state = 0

    # Initialize MDPs for each agent
    mdp1 = MDP(states, actions, transition_prob1, start_state)
    mdp2 = MDP(states, actions, transition_prob2, start_state)

    # Initialize agents
    agent1 = Agent(id=1, mdp=mdp1)
    agent2 = Agent(id=2, mdp=mdp2)

    # collect optimality gap stats for each sim
    nash_policies_opt_gaps_per_sim = []

    n_sim = 1000
    for i_sim in range(n_sim):
        g_vals = np.random.uniform(low=0, high=1, size=4)
        u1_vals = np.random.uniform(low=0, high=1, size=4)
        u2_vals = np.random.uniform(low=0, high=1, size=4)

        # Initialize seperable reward
        g_table = {}
        g_table[(0, 0)] = g_vals[0]
        g_table[(0, 1)] = g_vals[1]
        g_table[(1, 0)] = g_vals[2]
        g_table[(1, 1)] = g_vals[3]

        u1_table = {}
        u1_table[(0, 0)] = u1_vals[0]
        u1_table[(0, 1)] = u1_vals[1]
        u1_table[(1, 0)] = u1_vals[2]
        u1_table[(1, 1)] = u1_vals[3]

        u2_table = {}
        u2_table[(0, 0)] = u2_vals[0]
        u2_table[(0, 1)] = u2_vals[1]
        u2_table[(1, 0)] = u2_vals[2]
        u2_table[(1, 1)] = u2_vals[3]

        multi_agent = MultiAgent(agents=[agent1, agent2],
                                 multi_agent_reward=SeperableMultiAgentReward(num_agents=2,
                                                                              joint_static_term=g_table,
                                                                              independent_dynamic_terms=[u1_table, u2_table]))

        # Run Value Iteration - joint
        multi_agent.joint_value_iteration()

        # Run Value Iteration - decoupled
        # multi_agent.decoupled_value_iteration()

        # Run simulation
        n_steps = 100
        simulation = MultiAgentSimulation(multi_agent=multi_agent, max_steps=n_steps)

        # simulation.run_simulation()
        #
        # # Print results
        # n_agents = len(simulation.history)
        # for step in range(n_steps):
        #     print(f"Step {step:}")
        #     print(f" \t Reward: {simulation.joint_reward_history[step]}")
        #     for i, agent_hist_dict in enumerate(simulation.history):
        #         print(f" \t Agent {i}: State = {agent_hist_dict['states'][step]}, Action = {agent_hist_dict['actions'][step]}")
        #
        # welfare = simulation.calc_accumulated_reward()

        per_state_nash_policies = simulation.multi_agent.find_static_nash_policies()
        all_policies = simulation.multi_agent.get_all_deterministic_policies(states=simulation.multi_agent.get_joint_states(),
                                                                             actions=simulation.multi_agent.get_joint_actions())

        # all_normalized_value_funcs = [normalize_value_function(simulation.multi_agent.calc_value_function(p)) for p in all_policies]
        nash_policies = simulation.multi_agent.find_dynamic_nash_policies()

        # TODO - write functions for max global welfare find?

        nash_policies_joint_value_funcs = [simulation.multi_agent.calc_value_function(joint_policy=n_p) for n_p in nash_policies]

        # normalize value functions as we want to compare runs
        nash_opt_gaps = [simulation.multi_agent.calc_optimality_gap(n_p) for n_p in nash_policies]
        nash_policies_opt_gaps_per_sim.append(nash_opt_gaps)

        # TODO - collect statistics on random games regarding distance of nash policies from the global welfare
        nash_policies_decoupled_value_funcs = [simulation.multi_agent.calc_decoupled_value_function(joint_policy=n_p) for n_p in nash_policies]
        data_dict = {
            'sim_idx': i_sim,
            'simulation': simulation,
            'nash_policies': nash_policies,
            'nash_policies_decoupled_value_funcs': nash_policies_decoupled_value_funcs,
            'nash_policies_joint_value_funcs': nash_policies_joint_value_funcs,
            'nash_opt_gaps': nash_opt_gaps
        }

        if len(nash_policies) == 0:
            fname = os.path.join(os.getcwd(), '../sim_res/no_nash', f"sim_{i_sim}_no_nash.pkl")
            print(f"found game with no nash! saving {fname}")
            with open(fname, "wb") as f:
                pickle.dump(data_dict, f)

        elif simulation.multi_agent.optimal_policy not in nash_policies:
            fname = os.path.join(os.getcwd(), '../sim_res/global_is_not_nash', f"sim_{i_sim}_global_is_not_nash.pkl")
            print(f"Eureka! a game with non-Nash global optimum! saving {fname}")
            with open(fname, "wb") as f:
                pickle.dump(data_dict, f)
        else:
            fname = os.path.join(os.getcwd(), '../sim_res/regular', f"sim_{i_sim}_regular.pkl")
            print(f"regular sim! saving {fname}")
            with open(fname, "wb") as f:
                pickle.dump(data_dict, f)
        print(f"FINISHED SIM {i_sim}")

        # best_v = np.zeros_like(nash_policies_value_funcs[0])
        # for v in nash_policies_value_funcs:
        #     if np.all(v > best_v):
        #         best_v = np.array(v)
    # Prepare data for scatter plot
    simulation_ids = []  # Simulation index for each point
    gaps = []  # Optimality gap for each point

    for sim_idx, gaps_list in enumerate(nash_policies_opt_gaps_per_sim):
        simulation_ids.extend([sim_idx + 1] * len(gaps_list))  # Assign a simulation index to each gap
        gaps.extend(gaps_list)

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.scatter(simulation_ids, gaps, color='blue', alpha=0.6)
    plt.xlabel('Simulation Number')
    plt.ylabel('Optimality Gap')
    plt.title('Optimality Gaps for Nash Policies Across Simulations')
    plt.xticks(range(1, len(nash_policies_opt_gaps_per_sim) + 1))
    plt.show()
    # plt.figure()
    # plt.plot(np.arange(n_steps), welfare)
    # plt.show()