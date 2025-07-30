import copy

from tqdm import tqdm
import numpy as np
import os
import matplotlib
import pickle
import itertools
import random
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

from reward_functions import SeperableMultiAgentReward
from simulations.sim_utils import MDP, Agent, MultiAgent, MultiAgentSimulation


def init_multi_agent_setting(num_agents: int, num_states: int, num_actions: int) -> MultiAgent:
    assert num_agents > 1, "number of agents must be greater than 1"
    assert num_actions == num_states, "on our deterministic mapping case, state space and action space must have same size"

    trans_eye_mat = np.eye(num_states)
    states = set([i for i in range(num_states)])
    actions = set([i for i in range(num_actions)])

    # Define MDP parameters
    transition_prob1 = {
        s: {a: list(trans_eye_mat[a, :]) for a in actions}
        for s in states
    }
    start_state = 0

    # Initialize MDP
    mdp1 = MDP(states, actions, transition_prob1, start_state)

    # Initialize agents
    agents = []
    for i in range(num_agents):
        agents.append(Agent(id=i + 1, mdp=mdp1))

    u_list = []
    for _ in agents:
        u_vals = np.random.uniform(size=(num_states * num_actions))
        u_table = {}
        for i, s in enumerate(states):
            for j, a in enumerate(actions):
                u_table[s, a] = u_vals[(i) * num_states + j]
        u_list.append(u_table)

    # Generate all possible action combinations
    import itertools
    action_combinations = itertools.product(range(num_actions), repeat=num_agents)

    # Initialize seperable reward
    # Assign random values to each combination
    g_table = {tuple(actions): random.random() for actions in action_combinations}

    multi_agent = MultiAgent(agents=agents,
                             multi_agent_reward=SeperableMultiAgentReward(num_agents=num_agents,
                                                                          joint_static_term=g_table,
                                                                          independent_dynamic_terms=u_list))
    return multi_agent


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    num_agents = 2
    num_states = 3
    num_actions = num_states

    n_sim = 100
    for i_sim in tqdm(range(n_sim)):
        multi_agent = init_multi_agent_setting(num_agents=num_agents, num_states=num_states, num_actions=num_actions)

        # Run Value Iteration - joint
        multi_agent.joint_value_iteration()

        # Run Value Iteration - decoupled
        # multi_agent.decoupled_value_iteration()

        # Run simulation
        n_steps = 100
        simulation = MultiAgentSimulation(multi_agent=multi_agent, max_steps=n_steps)

        decoupled_nash_policies = simulation.multi_agent.find_dynamic_nash_policies(use_agent_decoupled_policies_only=True)
        buff_nash = simulation.multi_agent.find_buffered_decoupled_dynamic_nash_policies(buffer_size=2)
        # buff_policies, all_vs = simulation.multi_agent.calc_buffered_meaned_value_functions_all_policies(buffer_size=2)
        # TODO - write functions for max global welfare find?

        dec_nash_policies_joint_value_funcs = [
            simulation.multi_agent.calc_decoupled_value_function(joint_policy=n_p, use_for_global_value_calc=True)
            for n_p in decoupled_nash_policies
        ]
        buff_nash_policies_joint_value_funcs = [
            simulation.multi_agent.calc_decoupled_value_function(joint_policy=n_p, use_for_global_value_calc=True)
            for n_p in buff_nash
        ]

        data_dict = {
            'sim_idx': i_sim,
            'simulation': simulation,
            'dec_nash_policies': decoupled_nash_policies,
            'buff_nash_policies': buff_nash,
            'decoupled_nash_policies_joint_value_funcs': dec_nash_policies_joint_value_funcs,
            'buff_nash_policies_joint_value_funcs': buff_nash_policies_joint_value_funcs

        }

        if len(decoupled_nash_policies) == 0:
            fname = os.path.join(os.getcwd(), '../sim_res/updated_dec_and_buff/no_nash', f"sim_{i_sim}_no_nash.pkl")
            print(f"found game with no nash! saving {fname}")
            with open(fname, "wb") as f:
                pickle.dump(data_dict, f)

        elif simulation.multi_agent.optimal_policy not in decoupled_nash_policies:
            fname = os.path.join(os.getcwd(), '../sim_res/updated_dec_and_buff/global_is_not_nash', f"sim_{i_sim}_global_is_not_nash.pkl")
            print(f"Eureka! a game with non-Nash global optimum! saving {fname}")
            with open(fname, "wb") as f:
                pickle.dump(data_dict, f)
        else:
            fname = os.path.join(os.getcwd(), '../sim_res/updated_dec_and_buff/regular', f"sim_{i_sim}_regular.pkl")
            print(f"regular sim! saving {fname}")
            with open(fname, "wb") as f:
                pickle.dump(data_dict, f)
        print(f"FINISHED SIM {i_sim}")
