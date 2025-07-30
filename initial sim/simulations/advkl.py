import pickle
import os

# print(os.listdir('../sim_res/buff/global_is_not_nash'))

# with open('../sim_res/buff/global_is_not_nash/sim_107_global_opt_is_not_nash.pkl', 'rb') as f:
#     d = pickle.load(f)

with open('2_agents_3_states_uncoverable_sims.pkl', 'rb') as f:
    d = pickle.load(f)

for simulation in d.values():
    simulation.multi_agent.
x = 0

# d['decoupled_policies_joint_value_functions']
# best_dec_policy = min(d['decoupled_policies_joint_value_functions'], key=d['decoupled_policies_joint_value_functions'].get)