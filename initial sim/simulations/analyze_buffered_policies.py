import  pickle
import os
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np

from RL_utils import calc_policy_gap, calc_mean_policy_gap, calc_mean_normalized_policy_gap
from simulations.explore_peroids_in_opt_policy import visualize_joint_mdp


def strip_v_dict(v_dict):
    return {k: v[0] for k, v in v_dict.items()}


def get_minimal_gap_policy(policy_number_to_gap_dict):
    best_policy = min(policy_number_to_gap_dict, key=policy_number_to_gap_dict.get)
    return best_policy, policy_number_to_gap_dict[best_policy]


def get_policy_induced_transition_mapping(policy, is_buffered: bool = False):
    if is_buffered:
        return {(s, a): (s[1:] + (a,)) for s, a in policy.items()}
    else:
        return {(s, a): a for s, a in policy.items()}


def analyze_optimality_gap_for_nash_policies(data):
    is_data_updated = False

    # prepare all value functions precalculated data
    buff_v_funcs_dict = data['buff_policies_joint_value_functions']
    buff_v_funcs_dict = strip_v_dict(buff_v_funcs_dict)
    vrmean = {k: (v.reshape(4, 4)).mean(axis=0) for k, v in buff_v_funcs_dict.items()}

    dec_v_funcs_dict = data['decoupled_policies_joint_value_functions']
    dec_v_funcs_dict = strip_v_dict(dec_v_funcs_dict)

    # calc nash policies
    buff_nash = data.get('buffered_decoupled_nash_policies', None)
    if buff_nash is None:
        buff_nash = data['simulation'].multi_agent.find_buffered_decoupled_dynamic_nash_policies(buffer_size=2)
        data['buffered_decoupled_nash_policies'] = buff_nash
        is_data_updated = True

    dec_nash = data.get('decoupled_nash_policies', None)
    if dec_nash is None:
        dec_nash = data['simulation'].multi_agent.find_dynamic_nash_policies(use_agent_decoupled_policies_only=True)
        data['decoupled_nash_policies'] = dec_nash
        is_data_updated = True

    # get policy keys
    buff_nash_policy_numbers = [data['simulation'].multi_agent.get_policy_string_name(p) for p in buff_nash]
    dec_nash_policy_numbers = [data['simulation'].multi_agent.get_policy_string_name(p) for p in dec_nash]

    # create policy key - value function pairs
    buff_nash_v_dict = {k: vrmean[k] for k in buff_nash_policy_numbers}
    dec_nash_v_dict = {k: dec_v_funcs_dict[k] for k in dec_nash_policy_numbers}

    # calc gaps
    buff_nash_gaps_dict = {k: calc_mean_policy_gap(data['simulation'].multi_agent.value_function, v) for k, v
                           in buff_nash_v_dict.items()}
    dec_nash_gaps_dict = {k: calc_mean_policy_gap(data['simulation'].multi_agent.value_function, v) for k, v
                          in dec_nash_v_dict.items()}
    return buff_nash_gaps_dict, dec_nash_gaps_dict, is_data_updated


def analyze_optimality_gap(data):
    """
    This method collects optimality gap data from the optimal decoupled & buffered policies based on mean value function criterion
    :param data:
    :return:
    """
    is_data_updated = False
    is_improved_by_buffer = False
    buff_v_funcs_dict = data['buff_policies_joint_value_functions']
    buff_v_funcs_dict = strip_v_dict(buff_v_funcs_dict)
    vr = {k: v.reshape(4,4) for k, v in buff_v_funcs_dict.items()}
    vrmax = {k: (v.reshape(4, 4)).max(axis=0) for k, v in buff_v_funcs_dict.items()}
    vrmean = {k: (v.reshape(4, 4)).mean(axis=0) for k, v in buff_v_funcs_dict.items()}
    nash_policies = data.get('decoupled_nash_policies', None)
    if nash_policies is None:
        nash_policies = data['simulation'].multi_agent.find_dynamic_nash_policies(use_agent_decoupled_policies_only=True)
        data['decoupled_nash_policies'] = nash_policies
        is_data_updated = True

    dec_v_funcs_dict = data['decoupled_policies_joint_value_functions']
    dec_v_funcs_dict = strip_v_dict(dec_v_funcs_dict)

    # optimal policy analysis - check that buffered are not inferior ever

    # buff_gaps_dict = data['buff_mean_gaps_dict']
    buff_gaps_dict = {k: calc_mean_policy_gap(data['simulation'].multi_agent.value_function, vrm) for k, vrm in vrmean.items()}
    # dec_gaps_dict = data['decoupled_mean_gaps']
    dec_gaps_dict = {k: calc_mean_policy_gap(data['simulation'].multi_agent.value_function, v) for k, v in dec_v_funcs_dict.items()}

    min_buff_policy_number, min_buff_gap = get_minimal_gap_policy(buff_gaps_dict)
    min_dec_policy_number, min_dec_gap = get_minimal_gap_policy(dec_gaps_dict)

    if abs(min_buff_gap - min_dec_gap) < 1e-5:
        pass
        # print(f'buffer is not better than local. optimality gap is: {min_buff_gap}')
    elif min_dec_gap < min_buff_gap - 1e-5:
        # print(f'local dec is better than the buffered. seems like a bug or an interesting case. sim {data["sim_idx"]}')
        pass
    else:
        # print(f'found a case of improvement. sim: {data["sim_idx"]}')
        is_improved_by_buffer = True
        # best_buff_policy = data['buff_policies'][min_buff_policy_number]
        # best_dec_policy = data['decoupled_policies'][min_dec_policy_number]
        # opt_policy_induced_mapping = get_policy_induced_transition_mapping(data['simulation'].multi_agent.optimal_policy, is_buffered=False)
        # buff_policy_induced_mapping = get_policy_induced_transition_mapping(best_buff_policy, is_buffered=True)
        # dec_policy_induced_mapping = get_policy_induced_transition_mapping(best_dec_policy, is_buffered=False)
        # visualize_joint_mdp(opt_policy_induced_mapping)
        # visualize_joint_mdp(buff_policy_induced_mapping)
        # visualize_joint_mdp(dec_policy_induced_mapping)
        # plt.close('all')

    is_dec_nash_dec_opt = data['decoupled_policies'][min_dec_policy_number] in nash_policies

    return min_buff_gap, min_dec_gap, is_improved_by_buffer, is_dec_nash_dec_opt, is_data_updated


if __name__ == "__main__":
    data_dir = '../sim_res/buff/global_is_not_nash'

    buff_gaps = []
    dec_gaps = []
    improved_by_buffer = []
    buff_nash_gaps = []
    dec_nash_gaps = []
    for fname in tqdm(os.listdir(data_dir)):
        if 'sim' not in fname:
            continue
        file = os.path.join(data_dir, fname)

        with open(file, 'rb') as f:
            data = pickle.load(f)
        min_buff_gap, min_dec_gap, is_improved_by_buffer, is_dec_nash_dec_opt, is_data_updated = analyze_optimality_gap(data)
        # print(f"dec nash - file {fname} update status: {is_data_updated}")
        if is_data_updated:
            with open(file, 'wb') as f:
                pickle.dump(data, f)
        if not is_dec_nash_dec_opt:
            print(f"dec nash not dec opt. possible error. revisit {data['sim_idx']}")
        buff_gaps.append(min_buff_gap)
        dec_gaps.append(min_dec_gap)
        improved_by_buffer.append(is_improved_by_buffer)

        buff_nash_gaps_dict, dec_nash_gaps_dict, is_data_updated = analyze_optimality_gap_for_nash_policies(data)
        # print(f"buff nash - file {fname} update status: {is_data_updated}")
        print([v - min_buff_gap for v in buff_nash_gaps_dict.values()])
        if is_data_updated:
            with open(file, 'wb') as f:
                pickle.dump(data, f)
        buff_nash_gaps.append(list(buff_nash_gaps_dict.values()))
        dec_nash_gaps.append(list(dec_nash_gaps_dict.values()))

    with open('../sim_res/buff/gaps_stats_with_nash_data.pkl', 'wb') as f:
        pickle.dump(
            {
                'buff_gaps_normalized': buff_gaps,
                'dec_gaps_normalized': dec_gaps,
                'improved_by_buffer': improved_by_buffer,
                'buff_nash_gaps': buff_nash_gaps,
                'dec_nash_gaps': dec_nash_gaps,
            },
            f)




