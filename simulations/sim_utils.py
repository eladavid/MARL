from typing import List, Dict, Set, Sequence, Any
import numpy as np
import copy
from itertools import product

from RL_utils import calc_policy_gap
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

    # def value_iteration(self, theta: float = 1e-6):
    #     num_states = len(self.state_space)
    #     num_actions = len(self.action_space)
    #     while True:
    #         delta = 0
    #         for state_index in range(num_states):
    #             state = self.index_to_state(state_index)
    #             v = self.value_function[state_index]
    #             q_values = np.zeros(num_actions)
    #             for action_index in range(num_actions):
    #                 action = self.index_to_action(action_index)
    #
    #                 # TODO - implement to multiple agents separate mdps probs fetch
    #                 next_state_probs = self._mdp.transition_prob[state][action]
    #
    #                 reward = self._multi_agent_reward[tuple(state) + tuple(joint_action)]
    #
    #                 q_values[joint_action_index] = sum(next_state_probs[next_state_index] * (
    #                         reward + self.gamma * self.value_function[next_state_index])
    #                                                    for next_state_index in range(num_states))
    #             self.value_function[state_index] = np.max(q_values)
    #             self.policy[state_index] = np.unravel_index(np.argmax(q_values), num_actions)[0]
    #             delta = max(delta, abs(v - self.value_function[state_index]))
    #         if delta < theta:
    #             break


class MultiAgent:
    def __init__(self, agents: List[Agent], multi_agent_reward: Reward):
        self._multi_agent_reward = multi_agent_reward
        self._agents = agents

        self.reset_decision_making()
        # TODO - consider cases where different agents have different gamma values?
        self.gamma = 0.9

    @property
    def agents(self) -> List[Agent]:
        return self._agents

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
        state_space_size = 1
        for agent in self.agents:
            state_space_size *= len(agent._mdp.states)
        return state_space_size

    @property
    def action_space_size(self):
        action_space_size = 1
        for agent in self.agents:
            action_space_size *= len(agent._mdp.actions)
        return action_space_size

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

    def get_all_deterministic_single_agent_policies(self, agent_idx: int):
        states = list(self.agents[agent_idx].state_space)
        actions = list(self.agents[agent_idx].action_space)

        # Generate all permutations of actions for each joint state
        # Each permutation represents a different possible policy
        single_agent_policies = []

        for action_permutation in product(actions, repeat=len(states)):
            policy = dict(zip(states, action_permutation))
            single_agent_policies.append(policy)

        return single_agent_policies

    def extract_single_agent_policy_from_joint_policy(self, joint_policy: Dict[Any, Any], agent_idx: int) -> Dict[Any, Any]:
        single_agent_policy = {}
        for joint_state, joint_action in joint_policy.items():
            single_agent_policy[joint_state[agent_idx]] = joint_action[agent_idx]
        return single_agent_policy

    def inject_single_agent_policy_into_joint_policy(self,
                                                     joint_policy: Dict[Any, Any],
                                                     agent_idx: int,
                                                     single_agent_policy: Dict[Any, Any]) -> Dict[Any, Any]:
        # TODO - write test
        new_joint_agent_policy = copy.deepcopy(joint_policy)
        for joint_state, joint_action in joint_policy.items():
            new_joint_action = list(joint_policy[joint_state])
            new_joint_action[agent_idx] = single_agent_policy[joint_state[agent_idx]]
            new_joint_agent_policy[joint_state] = tuple(new_joint_action)
        return new_joint_agent_policy

    def get_all_single_agent_policy_alternatives(self, joint_policy, agent_idx):
        # TODO - write test
        # decompose single agent policy from joint-policy
        single_agent_policy = self.extract_single_agent_policy_from_joint_policy(joint_policy, agent_idx)

        # get all alternatives
        alt_policies = []
        states = list(self.agents[agent_idx].state_space)
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
        repeated_state_trans_probs = np.expand_dims(state_trans_probs, axis=tuple([idx for idx in range(n_agents) if idx != agent_idx]))
        # Todo - check for more than 2 agents
        repeated_state_trans_probs = np.repeat(repeated_state_trans_probs, repeats=n_agents, axis=0).flatten()

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
                value_function[state_index] = sum(next_joint_state_probs[next_state_idx] * (
                        curr_reward + self.gamma * value_function[next_state_idx])
                                                  for next_state_idx in range(len(value_function)))

                # update max difference between iterations
                delta = max(delta, abs(prev_v_estimate - value_function[state_index]))
            if delta < theta:
                break
        return value_function

    def joint_value_iteration(self, theta: float = 1e-6):
        # TODO - write test
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
                    q_values[joint_action_index] = sum(next_state_probs[next_state_index] * (
                                reward + self.gamma * self.value_function[next_state_index])
                                                       for next_state_index in range(num_states))

                # V(s) = max(Q(s, a))
                self.value_function[state_index] = np.max(q_values)

                # \pi(s) = argmax(Q(s, a))
                self.optimal_policy[state] = self.index_to_action(np.unravel_index(np.argmax(q_values), num_actions)[0])

                # convergence condition
                delta = max(delta, abs(v - self.value_function[state_index]))
            if delta < theta:
                break

        self.optimum_calculated = True

    def decoupled_value_iteration(self, theta: float = 1e-6):
        # TODO - revisit - this should not be used
        raise ValueError("THIS FUNCTION IS BROKEN")

        decoupled_value_function = [np.zeros(len(agent.state_space)) for i, agent in enumerate(self.agents)]
        decoupled_policies = [np.zeros(len(agent.state_space)) for i, agent in enumerate(self.agents)]

        num_states = self.state_space_size
        num_actions = self.action_space_size
        while True:
            delta = [0 for _ in range(len(decoupled_policies))]
            for state_index in range(num_states):
                joint_state = self.index_to_state(state_index)
                single_agent_states = [s for s in joint_state]

                v = [v[s] for v, s in zip(decoupled_value_function, single_agent_states)]
                decoupled_q_values = [np.zeros(len(agent.action_space)) for i, agent in enumerate(self.agents)]
                for joint_action_index in range(num_actions):

                    joint_action = self.index_to_action(joint_action_index)
                    reward = self._multi_agent_reward.get_reward(joint_state, joint_action)

                    single_agent_actions = [a for a in joint_action]
                    # TODO - implement to multiple agents separate mdps probs fetch

                    next_state_probs = [self.agents[i]._mdp.transition_prob[s][a]
                                        for i, (s, a) in enumerate(zip(single_agent_states, single_agent_actions))]

                    for i, (single_agent_q, state, action) in enumerate(zip(decoupled_q_values, single_agent_states, single_agent_actions)):
                        single_agent_q[action] = sum(next_state_probs[i][next_state_idx] * (reward + self.gamma * decoupled_value_function[i][next_state_idx])
                                                           for next_state_idx in range(len(decoupled_value_function[i])))

                for i, (single_agent_q, state) in enumerate(zip(decoupled_q_values, single_agent_states)):
                    decoupled_value_function[i][state] = np.max(single_agent_q)
                    decoupled_policies[i][state] = np.argmax(single_agent_q)
                    delta[i] = max(delta, abs(v[i] - decoupled_value_function[i][state]))

            if max(delta) < theta:
                break

    def calc_decoupled_value_function(self,
                                      joint_policy: Dict[Any, Any],
                                      theta: float = 1e-6) -> List[np.ndarray]:
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

        num_states = self.state_space_size
        decoupled_value_function = [np.zeros(num_states) for _ in self.agents]

        while True:
            delta = [0 for _ in range(len(decoupled_value_function))]
            for state_index in range(num_states):
                joint_state = self.index_to_state(state_index)

                # apply policy at current state
                joint_action = joint_policy[joint_state]

                # calc the partial reward per agent
                single_agent_rewards = [self._multi_agent_reward.get_single_agent_reward(agent_idx,
                                                                                         joint_state,
                                                                                         joint_action)
                                        for agent_idx, _ in enumerate(self.agents)]

                next_joint_state_probs = self.get_joint_transition_prob(joint_state, joint_action)

                # update V estimate
                for i, (single_agent_v, curr_reward) in enumerate(zip(decoupled_value_function, single_agent_rewards)):
                    prev_v_estimate = single_agent_v[state_index]

                    # update current V estimate
                    single_agent_v[state_index] = sum(next_joint_state_probs[next_state_idx] * (
                            curr_reward + self.gamma * single_agent_v[next_state_idx])
                                                      for next_state_idx in range(len(single_agent_v)))

                    # update max difference between iterations
                    delta[i] = max(delta[i], abs(prev_v_estimate - single_agent_v[state_index]))
            if max(delta) < theta:
                break
        return decoupled_value_function

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

    def find_dynamic_nash_policies(self):
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

        policies_dict = {
            hash_dict(policy_dict): policy_dict
            for policy_dict
            in self.get_all_deterministic_policies(states=self.get_joint_states(),
                                                   actions=self.get_joint_actions())
        }
        # calc once value function per policy
        policies_value_functions = {
            policy_hash: self.calc_decoupled_value_function(policy)
            for policy_hash, policy in policies_dict.items()
        }

        # for each policy - check if nash policy
        for policy_hash, policy in policies_dict.items():
            is_nash_policy = True
            # calculate each agent's value function
            agents_value_functions = policies_value_functions[policy_hash]

            # check if satisfies Nash condition on agent_idx coordinate
            for agent_idx, agent in enumerate(self.agents):
                # perform all possible single-agent policy alternatives and check value functions
                agent_i_alt_policies = self.get_all_single_agent_policy_alternatives(policy, agent_idx)

                # inject the single agent alternative into the joint policy
                alt_joint_policies = [self.inject_single_agent_policy_into_joint_policy(policy, agent_idx, sap)
                                      for sap in agent_i_alt_policies]

                # calculate alt policies value functions
                alt_policies_value_functions = [policies_value_functions[hash_dict(alt_joint_policy)]
                                                for alt_joint_policy in alt_joint_policies]

                # check if optimal for agent i
                is_optimal_for_agent_i = all(agents_value_functions[agent_idx][state_idx] >= alt_policy_value_functions[agent_idx][state_idx]
                                             for alt_policy_value_functions in alt_policies_value_functions
                                             for state_idx in range(self.state_space_size))

                # if not optimal for any agent - not nash
                if not is_optimal_for_agent_i:
                    is_nash_policy = False
                    break

            if is_nash_policy:
                nash_policies.append(policy)

        return nash_policies


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
