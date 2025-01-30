import pickle

with open("../sim_res/agent_decoupled_partition.pkl", "rb") as f:
    partition_dict = pickle.load(f)

opt_not_nash_list = partition_dict['opt_not_nash']

anomaly = opt_not_nash_list[109]

sim = anomaly['simulation']
sim.multi_agent.find_dynamic_nash_policies(use_agent_decoupled_policies_only=True)