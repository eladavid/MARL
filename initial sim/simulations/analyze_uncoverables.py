import pickle

from simulations.explore_peroids_in_opt_policy import get_policy_induced_transition_mapping, \
    build_policy_induced_mdp_graph, find_invalid_cycles, calculate_maximal_cycle, visualize_joint_mdp

with open('5_agents_2_states_uncoverable_sims.pkl', 'rb') as f:
    data = pickle.load(f)
count = 0
for multi_agent in data:
    policy_induced_mapping = get_policy_induced_transition_mapping(multi_agent, multi_agent.optimal_policy)
    # if policy_induced_mapping not in all_decoupled_policies:

    # max_length = calculate_maximal_cycle(policy_induced_mapping)
    # lens_cycles.append(max_length)
    # if max_length >= num_states:
    G = build_policy_induced_mdp_graph(policy_induced_mapping)
    invalid_cycles = find_invalid_cycles(G)
    if len(invalid_cycles) > 0:
        count += 1
    # max_length = calculate_maximal_cycle(policy_induced_mapping)
    # if max_length == 3:
    #     visualize_joint_mdp(policy_induced_mapping)
    # if max_length > len(multi_agent.agents[0].state_space):

print(count)
