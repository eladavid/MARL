from typing import List
import copy
import numpy as np

from reward_functions import SeperableMultiAgentReward
from simulations.sim_utils import MDP, Agent, MultiAgent

import pytest


# Define MDP parameters
@pytest.fixture(scope='module')
def states():
    return {0, 1}


@pytest.fixture(scope='module')
def actions():
    return {0, 1}


def transition_prob1():
    return {
        0: {0: [1., 0.], 1: [0., 1.]},
        1: {0: [1., 0.], 1: [0., 1.]}
    }


def mdp(states, actions, transition_prob, start_state: int):
    # Initialize MDPs for each agent
    return  MDP(states, actions, transition_prob, start_state)


def agent(id: int, mdp):
    # Initialize agents
    return Agent(id=id, mdp=mdp)


def g(agents_actions: List[int]):
    g_table = np.zeros((2, 2))
    g_table[0, 0] = 2
    g_table[0, 1] = 1
    g_table[1, 0] = 0
    g_table[1, 1] = 3
    return g_table[tuple(agents_actions)]


def u1(s1: int, a1: int):
    u1_table = np.zeros((2, 2))
    u1_table[0, 0] = 1
    u1_table[0, 1] = 3
    u1_table[1, 0] = 2
    u1_table[1, 1] = 1
    return u1_table[s1, a1]


def u2(s1: int, a1: int):
    u2_table = np.zeros((2, 2))
    u2_table[0, 0] = 1
    u2_table[0, 1] = -1
    u2_table[1, 0] = 2
    u2_table[1, 1] = 1
    return u2_table[s1, a1]


def two_agents_separable_reward(g, u1, u2):
    return SeperableMultiAgentReward(num_agents=2,
                                     joint_static_term=g,
                                     independent_dynamic_terms=[u1, u2])


def multi_agent_sep_reward(agent, two_agents_separable_reward):
    agent1 = copy.deepcopy(agent)
    agent2 = copy.deepcopy(agent)
    MultiAgent(agents=[agent1, agent2],
               multi_agent_reward=two_agents_separable_reward)

