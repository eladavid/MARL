from typing import List, Any

import numpy as np

# def joint_value_iteration(states: List[Any],
#                           actions: List[Any],
#                           transition_matrix: np.ndarray,
#                           theta: float = 1e-6):
#     num_states = len(states)
#     num_actions = len(actions)
#
#     value_function = np.zeros(num_states)
#     policy = {
#         s: actions[0] for s in states
#     }
#
#     while True:
#         delta = 0
#         for state_index, state in enumerate(states):
#             v = value_function[state_index]
#             q_values = np.zeros(num_actions)
#             for joint_action in actions:
#                 next_state_probs = get_joint_transition_prob(state, joint_action)
#
#                 reward = self._multi_agent_reward.get_reward(state, joint_action)
#
#                 # joint Q(s, a)
#                 q_values[joint_action_index] = sum(next_state_probs[next_state_index] * (
#                         reward + self.gamma * self.value_function[next_state_index])
#                                                    for next_state_index in range(num_states))
#
#             self.value_function[state_index] = np.max(q_values)
#             self.optimal_policy[state] = self.index_to_action(np.unravel_index(np.argmax(q_values), num_actions)[0])
#             delta = max(delta, abs(v - self.value_function[state_index]))
#         if delta < theta:
#             break


def normalize_value_function(value_function: np.ndarray):
    value_function = (value_function - value_function.min()) / (value_function.max() - value_function.min())
    return value_function


def calc_per_state_policy_gap(policy1_value_function, policy2_value_function):
    return [v_p1 - v_p2 for v_p1, v_p2 in zip(policy1_value_function, policy2_value_function)]


def calc_policy_gap(policy1_value_function, policy2_value_function):
    return max(calc_per_state_policy_gap(policy1_value_function, policy2_value_function))
