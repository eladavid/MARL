import copy
from typing import List, Dict, Set, Sequence

import numpy as np
import os
import matplotlib
import pickle

from RL_utils import normalize_value_function
from graph_utils import plot_policy_best_response_graph, compute_nash_convergence, find_nash_equilibrium_nodes
from reward_functions import Reward, SimpleTwoAgentsReward, SeperableMultiAgentReward
from simulations.sim_utils import MDP, Agent, MultiAgent, MultiAgentSimulation

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

import random


def get_policy_induced_transition_mapping(multi_agent: MultiAgent, policy: Dict):
    return {(s, a): multi_agent.index_to_state(
        int(np.where(multi_agent.get_joint_transition_prob(s, a))[0])) for s, a in policy.items()}


# visualize the mdp induced by the different decisions
import networkx as nx
import matplotlib.pyplot as plt


def visualize_joint_mdp(policy_induced_transition_prob):
    # Initialize a directed graph
    G = nx.DiGraph()

    # Add edges for joint state-action transitions
    for (joint_state, joint_action), next_state in policy_induced_transition_prob.items():
        # assuming deteremistic
        prob = 1.
        # Add an edge: current joint state -> next joint state
        G.add_edge(joint_state, next_state, action=joint_action, prob=prob)

    # Visualize the graph
    pos = nx.spring_layout(G)  # Layout for positioning nodes
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color="skyblue", font_size=10, font_weight="bold")

    # Draw edge labels (joint action and probability)
    edge_labels = {
        (u, v): f"A: {data['action']}\nP: {data['prob']:.2f}"
        for u, v, data in G.edges(data=True)
    }
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    # Draw directed edges explicitly without ambiguity
    nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=20, connectionstyle="arc3,rad=0.1")

    plt.title("Joint State-Action MDP Visualization")
    plt.show()

def build_policy_induced_mdp_graph(policy_induced_transition_prob):
    # Initialize a directed graph
    G = nx.DiGraph()

    # Add edges for joint state-action transitions
    for (joint_state, joint_action), next_state in policy_induced_transition_prob.items():
        # assuming deteremistic
        prob = 1.
        # Add an edge: current joint state -> next joint state
        G.add_edge(joint_state, next_state, action=joint_action, prob=prob)
    return G

def calculate_maximal_cycle(policy_induced_transition_prob):
    # Initialize a directed graph
    G = nx.DiGraph()

    # Add edges for joint state-action transitions
    for (joint_state, joint_action), next_state in policy_induced_transition_prob.items():
        # assuming deteremistic
        prob = 1.
        # Add an edge: current joint state -> next joint state
        G.add_edge(joint_state, next_state, action=joint_action, prob=prob)

    cycles = list(nx.simple_cycles(G))

    # Determine the maximum cycle length
    max_length = max(len(cycle) for cycle in cycles)

    # Find all cycles with maximum length
    maximal_cycles = [cycle for cycle in cycles if len(cycle) == max_length]

    # # Output the result
    # print(f"Number of maximal cycles: {len(maximal_cycles)}")
    # print(f"Maximal cycle length: {max_length}")
    return max_length

def is_valid_cycle(cycle):
    """
    Check if a cycle is valid.
    A cycle is valid if for each tuple index:
    - All values are the same, or
    - All values are different.
    """
    num_indices = len(cycle[0])  # Number of indices in tuple
    for i in range(num_indices):
        values = [node[i] for node in cycle]
        if len(set(values)) != 1 and len(set(values)) != len(values):
            return False
    return True


def find_invalid_cycles(graph):
    """
    Find invalid cycles in the graph.
    """
    invalid_cycles = []
    # Get all simple cycles in the graph
    cycles = list(nx.simple_cycles(graph))
    for cycle in cycles:
        if not is_valid_cycle(cycle):
            invalid_cycles.append(cycle)
    return invalid_cycles


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    num_agents_list = [7]
    num_states_list = [2]
    num_actions_list = [2]

    for num_agents, num_states, num_actions in zip(num_agents_list, num_states_list, num_actions_list):
        print(f"running for {num_agents} agents, {num_states} states")
        assert num_states == num_actions, "assuming one-to-one mapping from action to state"
        trans_eye_mat = np.eye(num_states)
        states = set([i for i in range(num_states)])
        actions = set([i for i in range(num_states)])

        uncoverables = []
        from tqdm import tqdm
        for i in tqdm(range(100)):
            # Example usage:

            # Define MDP parameters
            transition_prob1 = {
                s: {a: list(trans_eye_mat[a, :]) for a in actions}
                for s in states
            }
            single_agent_all_policies = None
            start_state = 0

            # Initialize MDPs for each agent
            mdp1 = MDP(states, actions, transition_prob1, start_state)
            mdp1_copy = copy.deepcopy(mdp1)
            # mdp2 = MDP(states, actions, transition_prob2, start_state)

            # Initialize agents
            agents = []
            for i in range(num_agents):
                agents.append(Agent(id=i+1, mdp=mdp1))

            # collect optimality gap stats for each sim
            nash_policies_opt_gaps_per_sim = []


            u_list = []
            for _ in agents:
                u_vals = np.random.uniform(size=(num_states * num_actions))
                u_table = {}
                for i, s in enumerate(states):
                    for j, a in enumerate(actions):
                        u_table[s,a] = u_vals[(i)*num_states+j]
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
            if not single_agent_all_policies:
                single_agent_all_policies = multi_agent.get_all_deterministic_policies(states, actions)
                # Generate all permutations of size num_agents
                permutations = itertools.product(single_agent_all_policies, repeat=num_agents)

                # Convert each permutation to a dict with tupled keys and tupled values
                all_decoupled_policies = []
                for perm in permutations:
                    combined_dict = {}
                    # Get all permutations of the keys across dictionaries
                    keys_permutations = itertools.product(*[d.keys() for d in perm])
                    for key_tuple in keys_permutations:
                        # Map key tuple to corresponding values
                        value_tuple = tuple(d[k] for d, k in zip(perm, key_tuple))
                        combined_dict[key_tuple] = value_tuple
                    all_decoupled_policies.append(combined_dict)

            # Run Value Iteration - joint
            multi_agent.joint_value_iteration()
            policy_induced_mapping = get_policy_induced_transition_mapping(multi_agent, multi_agent.optimal_policy)
            # if policy_induced_mapping not in all_decoupled_policies:

            # max_length = calculate_maximal_cycle(policy_induced_mapping)
            # lens_cycles.append(max_length)
            # if max_length >= num_states:
            G = build_policy_induced_mdp_graph(policy_induced_mapping)
            invalid_cycles = find_invalid_cycles(G)
            if len(invalid_cycles) > 0:
                print('eureka')
                # visualize_joint_mdp(policy_induced_mapping)
                uncoverables.append(multi_agent)


        # Run Value Iteration - decoupled
        # multi_agent.decoupled_value_iteration()
        import pickle
        with open(f'{num_agents}_agents_{num_states}_states_uncoverable_sims.pkl', 'wb') as f:
            pickle.dump(uncoverables, f)

