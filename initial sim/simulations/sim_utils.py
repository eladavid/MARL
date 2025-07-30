import itertools
from typing import List, Dict, Set, Sequence, Any, Optional
import numpy as np
import copy
from itertools import product
import networkx as nx
import math
from tqdm import tqdm

from RL_utils import calc_policy_gap
from graph_utils import find_nash_equilibrium_nodes, compute_nash_convergence
from reward_functions import Reward


def hash_dict(d):
    """Efficiently hash a dictionary by converting it to a tuple of sorted items."""
    return hash(frozenset(d.items()))


class MDP:
    def __init__(self,
                 states: Set[int],
                 actions: Set[int],
                 transition_prob: Dict[int, Dict[int, Sequence[float]]],
                 # rewards: Dict[int, Dict[int, float]],
                 start_state: int):
        self.states = states
        self.actions = actions
        self.transition_prob = transition_prob
        # self.rewards = rewards
        self._state = start_state

        self.validate_transition_prob()

    def validate_transition_prob(self):
        assert set(self.transition_prob.keys()) == self.states, "transition prob first dimension must cover the state space"
        for state_transition_prob in self.transition_prob.values():
            assert set(state_transition_prob.keys()) == self.actions
            for state_action_transition_prob in state_transition_prob.values():
                assert np.isclose(np.sum(state_action_transition_prob), 1.), 'transition matrix must be row stochastic'

    @property
    def state(self) -> int:
        return self._state

    def step(self, action: int):
        next_state_probs = self.transition_prob[self.state][action]
        next_state = np.random.choice(list(self.states), p=next_state_probs)
        # reward = self.rewards[self.state][action]
        self._state = next_state


class Agent:
    def __init__(self, id: int, mdp: MDP):
        self.id = id
        self._mdp = mdp  # assuming agent's action space is the one spanned by the mdp
        self._policy: np.ndarray = np.ones(len(mdp.actions)) / len(mdp.actions)  # init - random policy
        self._value_function = np.zeros((len(self.state_space),))

    @property
    def state_space(self) -> Set[int]:
        return self._mdp.states

    @property
    def action_space(self) -> Set[int]:
        # assuming agent's action space is the one spanned by the mdp
        return self._mdp.actions

    @property
    def policy(self) -> np.ndarray:
        return self._policy

    @property
    def value_function(self) -> np.ndarray:
        return self._value_function

    @policy.setter
    def policy(self, policy):
        assert len(policy) == len(self.action_space)
        self._policy = policy

    def index_to_state(self, state_index):
        return list(self.state_space)[state_index]

    def index_to_action(self, action_index):
        return list(self.action_space)[action_index]

    def select_action(self):
        return np.random.choice(list(self.action_space), p=self.policy)

    def curr_state(self) -> int:
        return self._mdp.state

    def act(self, action):
        self._mdp.step(action)


class MultiAgent:
    def __init__(self, agents: List[Agent], multi_agent_reward: Reward):
        self._multi_agent_reward = multi_agent_reward
        self._agents = agents

        self._state_space_size = None
        self._action_space_size = None

        self.reset_decision_making()
        # TODO - consider cases where different agents have different gamma values?
        self.gamma = 0.9

    @property
    def agents(self) -> List[Agent]:
        return self._agents

    @property
    def num_agents(self) -> int:
        return len(self.agents)

    @property
    def value_function(self):
        return self._value_function

    def select_action(self) -> List[int]:
        return [agent.select_action() for agent in self._agents]

    def act(self, actions: List[int]) -> float:
        states = []
        for i, agent in enumerate(self._agents):
            states.append(agent.curr_state())
            agent.act(action=actions[i])

        return self._multi_agent_reward.get_reward(states, actions)

    @property
    def state_space_size(self):
        # init
        if self._state_space_size is None:
            self._state_space_size = 1
            for agent in self.agents:
                self._state_space_size *= len(agent._mdp.states)
        # return
        return self._state_space_size

    @property
    def action_space_size(self):
        # init
        if self._action_space_size is None:
            self._action_space_size = 1
            for agent in self.agents:
                self._action_space_size *= len(agent._mdp.actions)
        # return
        return self._action_space_size

    def get_joint_states(self):
        return [self.index_to_state(state_idx) for state_idx in range(self.state_space_size)]

    def get_joint_actions(self):
        return [self.index_to_action(state_idx) for state_idx in range(self.action_space_size)]

    @staticmethod
    def get_all_deterministic_policies(states: Sequence, actions: Sequence):
        # TODO - write test
        joint_states = [s for s in states]
        joint_actions = [a for a in actions]
        # Generate all permutations of actions for each joint state
        # Each permutation represents a different possible policy
        policies = []

        for action_permutation in product(joint_actions, repeat=len(joint_states)):
            policy = dict(zip(joint_states, action_permutation))
            policies.append(policy)

        return policies

    @staticmethod
    def get_buffered_decoupled_policies(states: Sequence, actions: Sequence, buffer_size: int = 2):
        joint_states = [s for s in states]

        # buffer size = 1, should not have state tuple for history
        if not MultiAgent.is_tuple_of_tuples(joint_states[0]):
            last_joint_states_in_buffer = joint_states

        else:
            last_joint_states_in_buffer = list(set([s[-1] for s in joint_states]))
        joint_actions = [list(a) for a in actions]
        policies = []

        partitions = []
        num_agents = len(last_joint_states_in_buffer[0])
        for agent_idx in range(num_agents):
            partitions.append(MultiAgent.get_joint_states_partition_for_agent(joint_states, agent_idx))

        single_agent_actions = [list(set([a[i] for a in joint_actions])) for i in range(num_agents)]
        # prepare all permuations of actions along single agents states (state are generalized, i.e., include the buffer information)
        single_agent_permutations = [product(single_agent_actions[i], repeat=len(partitions[i].keys())) for i in range(num_agents)]

        # iterate over the cartesian product of all agent permutations
        for joint_action_permutation in product(*single_agent_permutations):
            joint_actions_for_policy = np.zeros((math.prod([len(tup) for tup in joint_action_permutation]), num_agents), dtype=int)
            for agent_idx in range(num_agents):
                agent_i_action = joint_action_permutation[agent_idx]
                for i, indices in enumerate(partitions[agent_idx].values()):
                    joint_actions_for_policy[indices, agent_idx] = agent_i_action[i]

            new_policy = dict(zip(joint_states, list(map(tuple, joint_actions_for_policy))))
            policies.append(new_policy)

        return policies

    @staticmethod
    def get_agent_decoupled_policies(all_deterministic_policies, use_buffered_states: bool = False):
        agent_decoupled_policies = []
        for p in all_deterministic_policies:
            is_agent_decoupled_policy = True
            tmp_state_action_mapping = {}
            prev_joint_states = {}
            for joint_state, joint_action in p.items():
                if use_buffered_states:
                    joint_state = joint_state[-1]
                    if prev_joint_states.get(joint_state, None):
                        continue
                    else:
                        prev_joint_states[joint_state] = 1
                for i, (single_agent_state, single_agent_action) in enumerate(zip(joint_state, joint_action)):
                    curr_action = tmp_state_action_mapping.get((i, single_agent_state), None)
                    # populate for the first time
                    if curr_action is None:
                        tmp_state_action_mapping[(i, single_agent_state)] = single_agent_action
                    # verify if action is consistent for same single agent state
                    elif curr_action != single_agent_action:
                        is_agent_decoupled_policy = False
                        break
                if not is_agent_decoupled_policy:
                    break
            if is_agent_decoupled_policy:
                agent_decoupled_policies.append(p)

        return agent_decoupled_policies

    def extract_single_agent_policy_from_joint_policy(self, joint_policy: Dict[Any, Any], agent_idx: int) -> Dict[Any, Any]:
        single_agent_policy = {}
        for joint_state, joint_action in joint_policy.items():
            single_agent_policy[joint_state] = joint_action[agent_idx]
        return single_agent_policy

    def inject_single_agent_policy_into_joint_policy(self,
                                                     joint_policy: Dict[Any, Any],
                                                     agent_idx: int,
                                                     single_agent_policy: Dict[Any, Any]) -> Dict[Any, Any]:
        # TODO - write test
        new_joint_agent_policy = copy.deepcopy(joint_policy)
        for joint_state, joint_action in joint_policy.items():
            new_joint_action = list(joint_policy[joint_state])
            new_joint_action[agent_idx] = single_agent_policy[joint_state]
            new_joint_agent_policy[joint_state] = tuple(new_joint_action)
        return new_joint_agent_policy

    def get_all_buffered_single_agent_policy_alternatives(self, joint_policy, agent_idx, partitions):
        # suggest all single agent's permutations:
        joint_actions = self.get_joint_actions()
        single_agent_actions = list(set([a[agent_idx] for a in joint_actions]))
        single_agent_permutations = product(single_agent_actions, repeat=len(partitions[agent_idx].keys()))

        buff_joint_states = list(joint_policy.keys())
        buff_joint_actions = list(joint_policy.values())
        policies = []
        for single_agent_action_perm in single_agent_permutations:
            # copy the orig policy
            alt_joint_actions = np.array(buff_joint_actions)
            # override the actions
            for i, indices in enumerate(partitions[agent_idx].values()):
                alt_joint_actions[indices, agent_idx] = single_agent_action_perm[i]

            new_policy = dict(zip(buff_joint_states, list(map(tuple, alt_joint_actions))))
            if new_policy != joint_policy:
                policies.append(new_policy)
        return policies

    def get_all_single_agent_policy_alternatives(self, joint_policy, agent_idx):
        # TODO - write test
        # decompose single agent policy from joint-policy
        single_agent_policy = self.extract_single_agent_policy_from_joint_policy(joint_policy, agent_idx)

        # get all alternatives
        alt_policies = []
        states = self.get_joint_states()
        actions = list(self.agents[agent_idx].action_space)

        # Generate all permutations of actions for each joint state
        # Each permutation represents a different possible policy
        for action_permutation in product(actions, repeat=len(states)):
            policy = dict(zip(states, action_permutation))
            if policy != single_agent_policy:
                alt_policies.append(policy)

        return alt_policies

    def reset_decision_making(self):
        joint_states = [self.index_to_state(state_idx) for state_idx in range(self.state_space_size)]
        tmp_action = self.index_to_action(0)
        self._value_function = np.zeros((self.state_space_size,))
        self.optimal_policy = {
            s: tmp_action for s in joint_states
        }
        self.optimum_calculated = False

    def index_to_state(self, idx):
        return np.unravel_index(idx, tuple([len(agent._mdp.states) for agent in self.agents]))

    def index_to_action(self, idx):
        return np.unravel_index(idx, tuple([len(agent._mdp.actions) for agent in self.agents]))

    def single_agent_trans_probs_to_joint_form(self, state_trans_probs: np.ndarray, agent_idx: int):
        n_agents = len(self.agents)
        state_space_dims_tuple = tuple([len(agent.state_space) for agent in self.agents])
        repeated_state_trans_probs = np.expand_dims(state_trans_probs, axis=tuple([idx for idx in range(n_agents) if idx != agent_idx]))

        repeated_state_trans_probs = np.broadcast_to(repeated_state_trans_probs, state_space_dims_tuple).flatten()
        return repeated_state_trans_probs

    def get_joint_transition_prob(self, joint_state, joint_action):
        joint_transition_prob = np.ones(self.state_space_size)
        for i, agent in enumerate(self.agents):
            agent_i_state_trans_probs = agent._mdp.transition_prob[joint_state[i]][joint_action[i]]
            # repeated
            repeated_state_trans_probs = self.single_agent_trans_probs_to_joint_form(np.array(agent_i_state_trans_probs), agent_idx=i)
            # agents' mdps are independent
            joint_transition_prob *= repeated_state_trans_probs
        return joint_transition_prob

    def calc_value_function(self,
                            joint_policy: Dict[Any, Any],
                            theta: float = 1e-6) -> List[np.ndarray]:

        num_states = self.state_space_size
        value_function = np.zeros(num_states)

        while True:
            delta = 0
            for state_index in range(num_states):
                joint_state = self.index_to_state(state_index)

                # apply policy at current state
                joint_action = joint_policy[joint_state]

                # calc the partial reward per agent
                curr_reward = self._multi_agent_reward.get_reward(joint_state, joint_action)

                next_joint_state_probs = self.get_joint_transition_prob(joint_state, joint_action)

                # update V estimate
                prev_v_estimate = value_function[state_index]

                # update current V estimate
                value_function[state_index] = sum(next_joint_state_probs * (curr_reward + self.gamma * value_function))

                # update max difference between iterations
                delta = max(delta, abs(prev_v_estimate - value_function[state_index]))
            if delta < theta:
                break
        return value_function

    def joint_value_iteration(self, theta: float = 1e-6):
        # TODO - write test
        if self.optimum_calculated:
            self.reset_decision_making()

        num_states = self.state_space_size
        num_actions = self.action_space_size
        while True:
            delta = 0
            for state_index in range(num_states):
                state = self.index_to_state(state_index)
                v = self.value_function[state_index]

                # tmp q values for current state, across all actions
                q_values = np.zeros(num_actions)

                # Q(s, a)
                for joint_action_index in range(num_actions):
                    joint_action = self.index_to_action(joint_action_index)

                    next_state_probs = self.get_joint_transition_prob(state, joint_action)

                    reward = self._multi_agent_reward.get_reward(state, joint_action)

                    # joint Q(s, a)
                    q_values[joint_action_index] = sum(next_state_probs * (reward + self.gamma * self.value_function))

                # V(s) = max(Q(s, a))
                self.value_function[state_index] = np.max(q_values)

                # \pi(s) = argmax(Q(s, a))
                self.optimal_policy[state] = self.index_to_action(np.unravel_index(np.argmax(q_values), num_actions)[0])

                # convergence condition
                delta = max(delta, abs(v - self.value_function[state_index]))
            if delta < self.num_agents * theta:
                break

        self.optimum_calculated = True

    def single_agent_decoupled_value_iteration(self,
                                               joint_policy: Dict,
                                               agent_idx: int,
                                               theta: float = 1e-6,
                                               use_global_reward: bool = False):
        # extract all other agents policies from the joint policy

        agent_i = self.agents[agent_idx]

        num_states = self.state_space_size
        joint_states = [self.index_to_state(state_idx) for state_idx in range(num_states)]

        # new impl
        agent_i_decoupled_value_function = np.zeros(num_states)

        decoupled_policy = copy.deepcopy(joint_policy)

        while True:
            delta = 0
            # iterate over all joint states
            for state_index in range(num_states):
                joint_state = self.index_to_state(state_index)
                single_agent_states = [s for s in joint_state]

                v = agent_i_decoupled_value_function[state_index]
                decoupled_q_values = np.zeros(len(agent_i.action_space))

                prev_joint_action = joint_policy[joint_state]
                # iterate over single agent actions given a fixed policy of other agents
                for agent_i_action in list(agent_i.action_space):
                    # construct the joint action based on all other agents' policy
                    new_joint_action = list(copy.deepcopy(prev_joint_action))
                    new_joint_action[agent_idx] = agent_i_action
                    new_joint_action = tuple(new_joint_action)

                    if not use_global_reward:
                        # calc single agent reward
                        agent_i_reward = self._multi_agent_reward.get_single_agent_reward(agent_idx,
                                                                                          joint_state,
                                                                                          new_joint_action)
                    else:
                        # calc global reward instead of single agent
                        agent_i_reward = self._multi_agent_reward.get_reward(joint_state, new_joint_action)

                    next_state_probs = self.get_joint_transition_prob(joint_state, new_joint_action)

                    # for i, (single_agent_q, state, action) in enumerate(zip(decoupled_q_values, single_agent_states, single_agent_actions)):
                    decoupled_q_values[agent_i_action] = sum(next_state_probs * (agent_i_reward + self.gamma * agent_i_decoupled_value_function))

                agent_i_decoupled_value_function[state_index] = np.max(decoupled_q_values)

                new_joint_action = list(copy.deepcopy(prev_joint_action))
                new_joint_action[agent_idx] = agent_i.index_to_action(np.argmax(decoupled_q_values))
                decoupled_policy[joint_state] = tuple(new_joint_action)
                delta = max(delta, abs(v - agent_i_decoupled_value_function[state_index]))

            if delta < theta:
                break
        return decoupled_policy, agent_i_decoupled_value_function

    def single_agent_decoupled_policies_decoupled_value_iteration(self,
                                                                  joint_policy: Dict,
                                                                  agent_idx: int,
                                                                  theta: float = 1e-6,
                                                                  use_global_reward: bool = False):
        """
        This function allows the agent to change his policies only in a way that is agnostic to other agents. meaning - cannot respond differently to different actions of other agents.

        :param joint_policy:
        :param agent_idx:
        :param theta:

        :return:
        """
        # extract all other agents policies from the joint policy

        agent_i = self.agents[agent_idx]

        num_joint_states = self.state_space_size
        all_joint_states = self.get_joint_states()
        single_agent_states = list(agent_i.state_space)

        # new impl
        agent_i_decoupled_value_function = np.zeros(num_joint_states)

        decoupled_policy = copy.deepcopy(joint_policy)

        while True:
            delta = 0
            # iterate over all joint states
            for state_index, single_agent_state in enumerate(single_agent_states):
                # collecting joint states containing this single agent state
                joint_states = [j_s for j_s in joint_policy.keys() if single_agent_state == list(j_s)[agent_idx]]

                # store previos values of all state containing single_agent_state
                v_per_j_s = [agent_i_decoupled_value_function[all_joint_states.index(j_s)] for j_s in joint_states]

                decoupled_q_values = np.zeros(len(agent_i.action_space))

                prev_joint_actions = [joint_policy[j_s] for j_s in joint_states]

                q_per_other_agents_states_dict = {idx: [] for idx in range(num_joint_states)}
                # iterate over single agent actions given a fixed policy of other agents
                for agent_i_action in list(agent_i.action_space):
                    q_per_other_agents_states = np.zeros(len(joint_states))
                    # construct the joint action based on all other agents' policy
                    for j_s_index, prev_joint_action in enumerate(prev_joint_actions):
                        new_joint_action = list(copy.deepcopy(prev_joint_action))
                        new_joint_action[agent_idx] = agent_i_action
                        new_joint_action = tuple(new_joint_action)

                        if not use_global_reward:
                            # calc single agent reward
                            agent_i_reward = self._multi_agent_reward.get_single_agent_reward(agent_idx,
                                                                                              joint_states[j_s_index],
                                                                                              new_joint_action)
                        else:
                            # calc global reward instead of single agent
                            agent_i_reward = self._multi_agent_reward.get_reward(joint_states[j_s_index], new_joint_action)

                        next_state_probs = self.get_joint_transition_prob(joint_states[j_s_index], new_joint_action)

                        # collect q-values over other agents' states. optimal action will be determined by best sum (same as best mean value - best expected return for single agent change)
                        q_per_other_agents_states[j_s_index] = sum(next_state_probs * (agent_i_reward + self.gamma * agent_i_decoupled_value_function))

                        canonical_joint_state_index = all_joint_states.index(joint_states[j_s_index])
                        q_per_other_agents_states_dict[canonical_joint_state_index].append(copy.deepcopy(q_per_other_agents_states[j_s_index]))

                    decoupled_q_values[agent_i_action] = np.sum(q_per_other_agents_states)

                single_agent_opt_action_idx = np.argmax(decoupled_q_values)
                # run over all joint states that contain current single agent state, collect q_max
                for joint_state_index, q_per_actions_list in q_per_other_agents_states_dict.items():
                    if len(q_per_actions_list) > 0:
                        agent_i_decoupled_value_function[joint_state_index] = q_per_actions_list[single_agent_opt_action_idx]

                for i, (prev_joint_action, j_s) in enumerate(zip(prev_joint_actions, joint_states)):
                    new_joint_action = list(copy.deepcopy(prev_joint_action))
                    new_joint_action[agent_idx] = agent_i.index_to_action(single_agent_opt_action_idx)
                    decoupled_policy[j_s] = tuple(new_joint_action)

                    canonical_joint_state_index = all_joint_states.index(j_s)
                    delta = max(delta, abs(v_per_j_s[i] - agent_i_decoupled_value_function[canonical_joint_state_index]))

            if delta < theta:
                break
        return decoupled_policy, agent_i_decoupled_value_function

    def initialize_decoupled_value_function(self, joint_policy, use_for_global_value_calc, mem_buffer_size):
        joint_states = list(joint_policy.keys())
        num_states = len(joint_states)
        decoupled_value_function = [np.zeros(num_states) for _ in
                                    self.agents] if not use_for_global_value_calc else [np.zeros(num_states)]
        if mem_buffer_size == 1:
            return decoupled_value_function
        else:
            for state_index, joint_state in enumerate(joint_states):
                # assuming determisitic mapping from state to action
                past_actions = joint_state[1:]
                if not use_for_global_value_calc:
                    # calc the partial reward per agent along all the buffer
                    single_agent_rewards = [
                        sum([(self.gamma ** i) * self._multi_agent_reward.get_single_agent_reward(agent_idx, j_s, j_a)
                         for i, (j_s, j_a) in enumerate(zip(joint_state[:-1], past_actions))])
                        for agent_idx, _ in enumerate(self.agents)]
                else:
                    single_agent_rewards = [
                        sum([(self.gamma ** i) * self._multi_agent_reward.get_reward(j_s, j_a) for i, (j_s, j_a) in
                         enumerate(zip(joint_state[:-1], past_actions))])]
                for single_agent_v, single_agent_reward in zip(decoupled_value_function, single_agent_rewards):
                    single_agent_v[state_index] = single_agent_reward
        return decoupled_value_function

    @staticmethod
    def is_tuple_of_tuples(variable):
        return isinstance(variable, tuple) and all(isinstance(item, tuple) for item in variable)

    def calc_decoupled_value_function(self,
                                      joint_policy: Dict[Any, Any],
                                      theta: float = 1e-6,
                                      use_for_global_value_calc: bool = False) -> List[np.ndarray]:
        # TODO - write test

        """
        This function calculates the "marginal"/"decoupled" value functions for all agents given a joint policy
        decoupled value function definition
        V^{\pi}_{i}(s) = E^{pi}[\sum_t{gamma^t * r_i(s, \pi)} | s0 = s]

        that is - expectation over the accumulated rewards *** received by player i ***
        """
        # Verify that 'reward_obj' has the required function
        if not (hasattr(self._multi_agent_reward, 'get_single_agent_reward') and
                callable(getattr(self._multi_agent_reward, 'get_single_agent_reward'))):
            raise AttributeError("The reward function does not have the required 'get_single_agent_reward'.")

        # extract mem buffer size assuming same size for all states...
        states_list = list(joint_policy.keys())
        joint_state_example = states_list[0]
        if self.is_tuple_of_tuples(joint_state_example):
            mem_buffer_size = len(joint_state_example)
            num_states = len(states_list)
        else:
            mem_buffer_size = 1
            num_states = self.state_space_size

        # # initialize value functions according to memory buffer
        # decoupled_value_function = self.initialize_decoupled_value_function(joint_policy, use_for_global_value_calc, mem_buffer_size)
        decoupled_value_function = [np.zeros(num_states) for _ in
                                    self.agents] if not use_for_global_value_calc else [np.zeros(num_states)]

        while True:
            delta = [0 for _ in range(len(decoupled_value_function))]

            for state_index, (joint_state, joint_action) in enumerate(joint_policy.items()):
                # make sure states are stored in list so common calc can go for both regular and mem buffered states
                if mem_buffer_size == 1:
                    joint_state = [joint_state]
                    joint_action = [joint_action]
                else:
                    joint_state = list(joint_state)
                    last_action = joint_action
                    joint_action = [s for s in joint_state[1:]]
                    joint_action.append(last_action)

                if not use_for_global_value_calc:
                    # single_agent_rewards = [
                    #     sum([(self.gamma ** i) * self._multi_agent_reward.get_single_agent_reward(agent_idx, j_s, j_a)
                    #          for i, (j_s, j_a) in enumerate(zip(joint_state, joint_action))])
                    #     for agent_idx, _ in enumerate(self.agents)]
                    single_agent_rewards = [self._multi_agent_reward.get_single_agent_reward(agent_idx,
                                                                                             joint_state[-1],
                                                                                             joint_action[-1])
                                            for agent_idx, _ in enumerate(self.agents)]
                else:
                    # single_agent_rewards = [
                    #     sum([(self.gamma ** i) * self._multi_agent_reward.get_reward(j_s, j_a) for i, (j_s, j_a) in
                    #          enumerate(zip(joint_state, joint_action))])]
                    single_agent_rewards = [self._multi_agent_reward.get_reward(joint_state[-1], joint_action[-1])
                                             for agent_idx, _ in enumerate(self.agents)]


                prob_vectors = [self.get_joint_transition_prob(joint_state[i], joint_state[i+1]) for i in range(len(joint_state) - 1)]
                prob_vectors.append(self.get_joint_transition_prob(joint_state[-1], joint_action[-1]))
                next_joint_state_probs = self.multi_step_transition_prob(prob_vectors)

                # update V estimate
                for i, (single_agent_v, curr_reward) in enumerate(zip(decoupled_value_function, single_agent_rewards)):
                    prev_v_estimate = single_agent_v[state_index]

                    # update current V estimate
                    single_agent_v[state_index] = np.dot(next_joint_state_probs,
                                                         curr_reward + self.gamma * single_agent_v)

                    # update max difference between iterations
                    delta[i] = max(delta[i], abs(prev_v_estimate - single_agent_v[state_index]))
            if np.max(delta) < theta:
                break
        return decoupled_value_function

    @staticmethod
    def multi_step_transition_prob(prob_vectors):
        """
        Compute the multi-step transition probability representation.

        :param prob_vectors: List of k probability vectors, each of length n
        :return: Flattened probability vector of size n^k
        """
        result = np.array(prob_vectors[0])  # Start with first probability vector
        for vec in prob_vectors[1:]:
            result = np.outer(result, vec).flatten()  # Compute outer product and flatten

        return result

    @staticmethod
    def calc_mean_buffered_value_function(buffered_policy, buffered_value_function):
        def get_indices_with_same_last_state(states):
            from collections import defaultdict
            groups = defaultdict(list)

            # Iterate and group indices
            for idx, tup in enumerate(states):
                groups[tup[-1]].append(idx)

            # Convert to list of lists
            result = list(groups.values())
            return result
        # average along state with same last state across different past buffers
        indices = get_indices_with_same_last_state(list(buffered_policy.keys()))
        return buffered_value_function[indices].mean(axis=1)

    def calc_optimality_gap(self, alt_policy):
        # if optimal policy is not known - calculate it
        if not self.optimum_calculated:
            self.joint_value_iteration()
        alt_value_function = self.calc_value_function(alt_policy)

        return calc_policy_gap(self.value_function, alt_value_function)

    def find_static_nash_policies(self):
        """
        This function maps state to its "Static Nash" policies
        Static Nash - as if the system works in "open loop" such that actions are taken but the state is frozen
        :return: dict[state, List[nash policies]]
        """
        # Verify that 'reward_obj' has the required function
        if not (hasattr(self._multi_agent_reward, 'get_single_agent_reward') and
                callable(getattr(self._multi_agent_reward, 'get_single_agent_reward'))):
            raise AttributeError("The reward function does not have the required 'get_single_agent_reward'.")

        static_nash_policies_per_state = {}
        for state_idx in range(self.state_space_size):
            state = self.index_to_state(state_idx)
            state_nash_policies = self.find_state_static_nash_policies(state)
            static_nash_policies_per_state[state_idx] = state_nash_policies
        return static_nash_policies_per_state

    def find_state_static_nash_policies(self, joint_state):
        """
        Finds Static Nash equilibrium policies for a given state
        Static Nash - as if the system works in "open loop" such that actions are taken but the state is frozen

        Parameters:
        - joint_state: The current state for which to find Nash policies.

        Returns:
        - static_nash_policies: A list of Nash equilibrium policies for each agent.
        """
        static_nash_policies = []

        num_actions = self.action_space_size
        # TODO - add function that extract feasible actions per state
        for action_idx in range(num_actions):
            joint_action = self.index_to_action(action_idx)

            is_nash_equilibrium = True
            for agent_idx, agent in enumerate(self.agents):
                agent_reward = self._multi_agent_reward.get_single_agent_reward(agent_index=agent_idx,
                                                                                agents_states=joint_state,
                                                                                agents_actions=joint_action)
                # Check if there's an incentive to deviate for this agent
                for alt_action in list(self.agents[agent_idx].action_space):
                    if alt_action != joint_action[agent_idx]:
                        # Modify the joint action for the alternative action
                        alternative_joint_action = list(joint_action)
                        alternative_joint_action[agent_idx] = alt_action
                        alternative_joint_action = tuple(alternative_joint_action)

                        # Compare the reward of deviating vs staying with joint_action
                        if self._multi_agent_reward.get_single_agent_reward(agent_index=agent_idx,
                                                                            agents_states=joint_state,
                                                                            agents_actions=alternative_joint_action) > agent_reward:
                            is_nash_equilibrium = False
                            break

                if not is_nash_equilibrium:
                    break

            if is_nash_equilibrium:
                static_nash_policies.append(joint_action)

        return static_nash_policies

    def find_dynamic_nash_policies(self, use_agent_decoupled_policies_only: bool = False):
        # TODO - write test
        # TODO - optimize
        """
        This function maps state to its "Dynamic Nash" policies
        Dynamic Nash - the game's actual Nash (system works in "closed loop").
        :return: dict[state, List[nash policies]]
        """
        # Verify that 'reward_obj' has the required function
        if not (hasattr(self._multi_agent_reward, 'get_single_agent_reward') and
                callable(getattr(self._multi_agent_reward, 'get_single_agent_reward'))):
            raise AttributeError("The reward function does not have the required 'get_single_agent_reward'.")

        nash_policies = []
        joint_states = self.get_joint_states()
        if not use_agent_decoupled_policies_only:
            policies_list = self.get_all_deterministic_policies(states=joint_states,
                                                                actions=self.get_joint_actions())
        # include only agent decoupled policies
        else:
            # policies_list = self.get_agent_decoupled_policies(policies_list)
            policies_list = self.get_buffered_decoupled_policies(joint_states, self.get_joint_actions(), buffer_size=1)
        # create policy dict
        policies_dict = {
            self.get_policy_string_name(policy_dict): policy_dict
            for policy_dict
            in policies_list
        }

        # calc once value function per policy
        policies_value_functions = {
            policy_number: self.calc_decoupled_value_function(policy, theta=1e-6)
            for policy_number, policy in tqdm(policies_dict.items())
        }

        # if checking only agent-decoupled policies - nash definition changes as same action must be applied along all
        #                                             states with same "single agent marginal state"
        # thus, we calculate mean value function across all such states, which stands for assuming uniform initial state distribution
        if use_agent_decoupled_policies_only:
            partitions = []
            for agent_idx in range(self.num_agents):
                partitions.append(self.get_joint_states_partition_for_agent(joint_states, agent_idx))

            def calc_single_agent_decoupled_policy_expected_value_func(all_agents_partitions, value_function, agent_idx):
                agent_partition = all_agents_partitions[agent_idx]
                meaned_value_function = np.zeros(len(agent_partition.keys()))
                for key, indices in agent_partition.items():
                    meaned_value_function[key] = value_function[indices].mean()  # Compute mean over indices in the partition
                return meaned_value_function

            for agents_decoupled_value_functions in policies_value_functions.values():
                for i, value_function in enumerate(agents_decoupled_value_functions):
                    meand_value_function = calc_single_agent_decoupled_policy_expected_value_func(partitions,
                                                                                                  value_function,
                                                                                                  agent_idx=i)
                    agents_decoupled_value_functions[i] = meand_value_function

        # for each policy - check if nash policy
        for policy_number, policy in policies_dict.items():
            is_nash_policy = True
            # calculate each agent's value function
            agents_value_functions = policies_value_functions[policy_number]

            # check if satisfies Nash condition on agent_idx coordinate
            for agent_idx, agent in enumerate(self.agents):
                # perform all possible single-agent policy alternatives and check value functions
                if use_agent_decoupled_policies_only:
                    alt_joint_policies = self.get_all_buffered_single_agent_policy_alternatives(policy, agent_idx,
                                                                                                    partitions)
                else:
                    agent_i_alt_policies = self.get_all_single_agent_policy_alternatives(policy, agent_idx)

                    # inject the single agent alternative into the joint policy
                    alt_joint_policies = [self.inject_single_agent_policy_into_joint_policy(policy, agent_idx, sap)
                                          for sap in agent_i_alt_policies]

                # calculate alt policies value functions
                alt_policies_value_functions = [policies_value_functions[self.get_policy_string_name(alt_joint_policy)]
                                                for alt_joint_policy in alt_joint_policies]

                # check if optimal for agent i
                is_optimal_for_agent_i = np.all([agents_value_functions[agent_idx].mean() + 1e-6 >= alt_policy_value_functions[agent_idx].mean()
                                                 for alt_policy_value_functions in alt_policies_value_functions])

                # if not optimal for any agent - not nash
                if not is_optimal_for_agent_i:
                    is_nash_policy = False
                    break

            if is_nash_policy:
                nash_policies.append(policy)

        return nash_policies

    def build_policies_best_response_graph(self, use_agent_decoupled_policies_only: bool = False):
        agent_colors = ['red', 'blue', 'green', 'purple', 'orange']
        assert len(self.agents) <= len(agent_colors), f"cannot work with more agents than {len(agent_colors)}"
        policies_list = self.get_all_deterministic_policies(states=self.get_joint_states(),
                                                            actions=self.get_joint_actions())
        if use_agent_decoupled_policies_only:
            policies_list = self.get_agent_decoupled_policies(policies_list)

        nash_policies = self.find_dynamic_nash_policies(use_agent_decoupled_policies_only=use_agent_decoupled_policies_only)
        # Create a directed graph
        policy_graph = nx.MultiDiGraph()

        # Add nodes (policies)
        for p in policies_list:
            p_number = self.get_policy_string_name(p)
            policy_graph.add_node(p_number, color=agent_colors[2] if p in nash_policies else agent_colors[1])

        # build best-response mapping for all agents
        for i, _ in enumerate(self.agents):
            # Add directed edges based on best responses
            for policy in policies_list:
                # Add edge for agent i's best response

                # special value iteration case
                if use_agent_decoupled_policies_only:
                    agent_i_best_response, _  = self.single_agent_decoupled_policies_decoupled_value_iteration(joint_policy=policy, agent_idx=i)
                # value iteration over a large span of policies (faster than brute force ? not necessarily)
                else:
                    agent_i_best_response, _ = self.single_agent_decoupled_value_iteration(joint_policy=policy, agent_idx=i)
                policy_graph.add_edge(self.get_policy_string_name(policy),
                                      self.get_policy_string_name(agent_i_best_response),
                                      agent=f"agent {i+1}", color=agent_colors[i], style='solid' if i == 0 else 'dashed', weight=i+1)
        return policy_graph

    def build_nash_convergence_graph(self, policy_best_response_graph):
        agent_colors = ['red', 'blue', 'green', 'purple', 'orange']
        assert len(self.agents) <= len(agent_colors), f"cannot work with more agents than {len(agent_colors)}"
        nash_nodes = find_nash_equilibrium_nodes(policy_best_response_graph)
        agents_list = sorted(list(set([data.get("agent", None) for _, _, data in policy_best_response_graph.edges(data=True)])))
        per_node_nash_convergence = compute_nash_convergence(policy_best_response_graph, nash_nodes, agents_list)

        # Create a directed graph
        nash_convegence_graph = nx.MultiDiGraph()

        # duplicate nodes for all policies
        for node, attributes in policy_best_response_graph.nodes(data=True):
            nash_convegence_graph.add_node(node, **attributes)

        # connect edges based on nash convergence
        for policy_node, per_agent_act_convergence in per_node_nash_convergence.items():
            for i, agent_i_response_nash in enumerate(per_agent_act_convergence):
                nash_convegence_graph.add_edge(policy_node,
                                               agent_i_response_nash,
                                               agent=f"agent {i+1}", color=agent_colors[i], style='solid' if i == 0 else 'dashed', weight=i+1)

        return nash_convegence_graph

    @staticmethod
    # def get_policy_string_name(policy):
    #     joint_values = [v for v in policy.values()]
    #     p_name = ''.join(str(x) for values in joint_values for x in values)
    #     p_number = int(p_name, base=2)
    #     return p_number
    def get_policy_string_name(policy):
        joint_values = tuple(tuple(v) for v in policy.values())  # Ensure hashability
        return hash(joint_values)

    @staticmethod
    def get_joint_states_partition_for_agent(joint_states, agent_idx):
        from collections import defaultdict

        # find which states should be "mean"ed
        partition = defaultdict(list)
        for index, tup in enumerate(joint_states):
            if MultiAgent.is_tuple_of_tuples(tup):
                multi_step_state = list(tup)
                value = tuple([s[agent_idx] for s in multi_step_state])
            else:
                value = tup[agent_idx]
            # value = tup[agent_idx]
            partition[value].append(index)
        return partition

    def find_buffered_decoupled_dynamic_nash_policies(self, buffer_size: int = 2, precalculated_value_functions_dict = None):
        # TODO - write test
        # TODO - optimize
        """
        This function maps state to its "Dynamic Nash" policies
        Dynamic Nash - the game's actual Nash (system works in "closed loop").
        :return: dict[state, List[nash policies]]
        """
        # Verify that 'reward_obj' has the required function
        if not (hasattr(self._multi_agent_reward, 'get_single_agent_reward') and
                callable(getattr(self._multi_agent_reward, 'get_single_agent_reward'))):
            raise AttributeError("The reward function does not have the required 'get_single_agent_reward'.")

        nash_policies = []
        # span the buffered state & action space
        joint_buffered_states = list(itertools.product(self.get_joint_states(), repeat=buffer_size)) if buffer_size > 1 else self.get_joint_states()

        if buffer_size > 1:
            policies_list = self.get_buffered_decoupled_policies(
                states=joint_buffered_states,
                actions=self.get_joint_actions(),
                buffer_size=buffer_size,
            )
        else:
            policies_list = self.get_agent_decoupled_policies(self.get_all_deterministic_policies(joint_buffered_states, self.get_joint_actions()))

        # create policy dict
        policies_dict = {
            self.get_policy_string_name(policy_dict): policy_dict
            for policy_dict
            in policies_list
        }

        # TODO - perform all the buffered calculation...

        if precalculated_value_functions_dict is None:
            # calc once value function per policy
            policies_value_functions = {
                policy_number: self.calc_decoupled_value_function(policy, theta=1e-6)
                for policy_number, policy in tqdm(policies_dict.items())
            }
        else:
            policies_value_functions = {
                policy_number: value_function for policy_number, value_function in precalculated_value_functions_dict.items()
            }

        # if checking only agent-decoupled policies - nash definition changes as same action must be applied along all
        #                                             states with same "single agent marginal state"
        # thus, we calculate mean value function across all such states, which stands for assuming uniform initial state distribution
        partitions = []
        for agent_idx in range(self.num_agents):
            partitions.append(self.get_joint_states_partition_for_agent(joint_buffered_states, agent_idx))

        def calc_single_agent_decoupled_policy_expected_value_func(all_agents_partitions, value_function, agent_idx):
            agent_partition = all_agents_partitions[agent_idx]
            meaned_value_function = np.zeros(len(agent_partition.keys()))
            for idx, indices in enumerate(agent_partition.values()):
                meaned_value_function[idx] = value_function[indices].mean()  # Compute mean over indices in the partition
            return meaned_value_function

        for agents_decoupled_value_functions in policies_value_functions.values():
            for i, value_function in enumerate(agents_decoupled_value_functions):
                meand_value_function = calc_single_agent_decoupled_policy_expected_value_func(partitions,
                                                                                              value_function,
                                                                                              agent_idx=i)
                agents_decoupled_value_functions[i] = meand_value_function

        # for each policy - check if nash policy
        for policy_number, policy in policies_dict.items():
            is_nash_policy = True
            # calculate each agent's value function
            agents_value_functions = policies_value_functions[policy_number]

            # check if satisfies Nash condition on agent_idx coordinate
            for agent_idx, agent in enumerate(self.agents):
                # perform all possible single-agent policy alternatives and check value functions
                alt_joint_policies = self.get_all_buffered_single_agent_policy_alternatives(policy, agent_idx, partitions)

                # calculate alt policies value functions
                alt_policies_value_functions = [policies_value_functions[self.get_policy_string_name(alt_joint_policy)]
                                                for alt_joint_policy in alt_joint_policies]

                # check if optimal for agent i
                is_optimal_for_agent_i = np.all([agents_value_functions[agent_idx].mean() + 1e-6 >= alt_policy_value_functions[agent_idx].mean()
                                                 for alt_policy_value_functions in alt_policies_value_functions])

                # if not optimal for any agent - not nash
                if not is_optimal_for_agent_i:
                    is_nash_policy = False
                    break

            if is_nash_policy:
                nash_policies.append(policy)

        return nash_policies

    def calc_buffered_value_functions_all_policies(self, buffer_size: int = 2):
        # Verify that 'reward_obj' has the required function
        if not (hasattr(self._multi_agent_reward, 'get_single_agent_reward') and
                callable(getattr(self._multi_agent_reward, 'get_single_agent_reward'))):
            raise AttributeError("The reward function does not have the required 'get_single_agent_reward'.")

        nash_policies = []
        # span the buffered state & action space
        joint_buffered_states = list(itertools.product(self.get_joint_states(), repeat=2))

        policies_list = self.get_buffered_decoupled_policies(
            states=joint_buffered_states,
            actions=self.get_joint_actions(),
            buffer_size=buffer_size,
        )

        # create policy dict
        policies_dict = {
            self.get_policy_string_name(policy_dict): policy_dict
            for policy_dict
            in policies_list
        }

        # TODO - perform all the buffered calculation...
        # calc once value function per policy
        from tqdm import tqdm
        policies_value_functions = {
            policy_number: self.calc_decoupled_value_function(policy, theta=1e-8, use_for_global_value_calc=True)
            for policy_number, policy in tqdm(policies_dict.items())
        }

        # if checking only agent-decoupled policies - nash definition changes as same action must be applied along all
        #                                             states with same "single agent marginal state"
        # thus, we calculate mean value function across all such states, which stands for assuming uniform initial state distribution
        #partitions = []
        #for agent_idx in range(self.num_agents):
        #    partitions.append(self.get_joint_states_partition_for_agent(joint_buffered_states, agent_idx))

        #def calc_single_agent_decoupled_policy_expected_value_func(all_agents_partitions, value_function, agent_idx):
        #    agent_partition = all_agents_partitions[agent_idx]
        #    meaned_value_function = np.zeros(len(agent_partition.keys()))
        #    for idx, indices in enumerate(agent_partition.values()):
        #        meaned_value_function[idx] = value_function[
        #            indices].mean()  # Compute mean over indices in the partition
        #    return meaned_value_function

        #for agents_decoupled_value_functions in policies_value_functions.values():
        #    for i, value_function in enumerate(agents_decoupled_value_functions):
        #        meand_value_function = calc_single_agent_decoupled_policy_expected_value_func(partitions,
        #                                                                                      value_function,
        #                                                                                      agent_idx=i)
        #        agents_decoupled_value_functions[i] = meand_value_function

        return policies_dict, policies_value_functions


class MultiAgentSimulation:
    def __init__(self, multi_agent: MultiAgent, max_steps):
        self.multi_agent = multi_agent
        self.max_steps = max_steps
        self.step_counter = 0
        self.history = [{
            'states': [],
            'actions': [],
            'rewards': []
        } for _ in self.multi_agent.agents]
        self.joint_reward_history = []

    def reset(self):
        self.step_counter = 0
        self.joint_reward_history.clear()
        for agent_history_dict in self.history:
            agent_history_dict['states'].clear()
            agent_history_dict['actions'].clear()
            agent_history_dict['rewards'].clear()

    @staticmethod
    def update_agent_history(agent_history_dict: Dict, curr_state: int, action: int):
        agent_history_dict['states'].append(curr_state)
        agent_history_dict['actions'].append(action)

    def run_step(self):
        actions = self.multi_agent.select_action()
        multi_agent_reward = self.multi_agent.act(actions)

        # update each agent's history of state & action
        for i, agent in enumerate(self.multi_agent.agents):
            curr_state = self.multi_agent.agents[i].curr_state()
            self.update_agent_history(self.history[i], curr_state, actions[i])

        self.joint_reward_history.append(multi_agent_reward)

        self.step_counter += 1

    def run_simulation(self):
        self.reset()
        while self.step_counter < self.max_steps:
            self.run_step()

    def calc_accumulated_reward(self):
        accum_reward = np.cumsum(self.joint_reward_history)
        return accum_reward
