from abc import ABC, abstractmethod
from typing import List, Callable, Dict, Tuple, Any

import numpy as np


class Reward(ABC):
    def __init__(self, num_agents):
        self._num_agents = num_agents

    @property
    def num_agents(self):
        return self._num_agents

    @abstractmethod
    def get_reward(self, agents_states: List, agents_actions: List):
        pass


class SimpleTwoAgentsReward(Reward):
    """
    Simple hard-coded tabular reward for 2 agents, 2 states, 2 actions.
    """
    def __init__(self):
        super().__init__(2)

        # table for 2 agents with values r(s1, s2, a1, a2)
        self._multi_agent_reward = np.zeros((self.num_agents, self.num_agents, self.num_agents, self.num_agents))
        self.fill_reward_table()

    def fill_reward_table(self):
        # joint s = [0, 0]
        self._multi_agent_reward[0, 0, 0, 0] = 1.
        self._multi_agent_reward[0, 0, 0, 1] = 0.
        self._multi_agent_reward[0, 0, 1, 0] = 0.
        self._multi_agent_reward[0, 0, 1, 1] = 1.
        # joint s = [0, 1]
        self._multi_agent_reward[0, 1, 0, 0] = 0.
        self._multi_agent_reward[0, 1, 0, 1] = 1.
        self._multi_agent_reward[0, 1, 1, 0] = 1.
        self._multi_agent_reward[0, 1, 1, 1] = 0.
        # joint s = [1, 0]
        self._multi_agent_reward[1, 0, 0, 0] = 1.
        self._multi_agent_reward[1, 0, 0, 1] = 1.
        self._multi_agent_reward[1, 0, 1, 0] = 1.
        self._multi_agent_reward[1, 0, 1, 1] = 0.
        # joint s = [1, 1]
        self._multi_agent_reward[1, 1, 0, 0] = 1.
        self._multi_agent_reward[1, 1, 0, 1] = 0.
        self._multi_agent_reward[1, 1, 1, 0] = 0.
        self._multi_agent_reward[1, 1, 1, 1] = 1.

    def get_reward(self, agents_states: List, agents_actions: List):
        return self._multi_agent_reward[tuple(agents_states) + tuple(agents_actions)]


class SeperableMultiAgentReward(Reward):
    """
    This class implements multi-agent reward with the following form:
               r_i = g(a_1,...,a_n) + u_i(s_i, a_i)
     The global welfare can be formulated as
               R = g(a_1,...,a_n) + sum(u_j(s_j, a_j)))
    """
    def __init__(self, num_agents: int,
                 joint_static_term: Dict[Tuple[Any, ...], float],
                 independent_dynamic_terms: List[Dict[Tuple[Any, Any], float]]):
        assert num_agents == len(independent_dynamic_terms), "len(independent_dynamic_terms) must match number of agents"
        super().__init__(num_agents)
        self._g = joint_static_term
        self._u_list = independent_dynamic_terms

    def get_reward(self, agents_states: List, agents_actions: List):
        r = self._g[agents_actions]
        for i, u_i in enumerate(self._u_list):
            r += u_i[agents_states[i], agents_actions[i]]
        return r

    def get_single_agent_reward(self, agent_index: int, agents_states: List, agents_actions: List):
        # return r_i = g(a_1,...,a_n) + u_i(s_i, a_i)
        return self._g[agents_actions] + self._u_list[agent_index][agents_states[agent_index], agents_actions[agent_index]]

    @staticmethod
    def calc_distance_between_rewards(separable_reward1,
                                      separable_reward2):
        assert len(separable_reward1._u_list) == len(separable_reward1._u_list), "distance is defined for rewards with the same number of agents"
        g_dist = np.abs(np.array(list(separable_reward1._g.values())) - np.array(list(separable_reward2._g.values())))
        u_dist = np.abs(sum([np.array(list(u_1_i.values())) - np.array(list(u_2_i.values()))
                             for u_1_i, u_2_i in zip(separable_reward1._u_list, separable_reward2._u_list)]))
        return np.sum(u_dist + g_dist)

    def get_reward_as_vector(self):
        vec = [list(u_i.values()) for u_i in self._u_list]
        vec.append(list(self._g.values()))
        return np.concatenate(vec)



