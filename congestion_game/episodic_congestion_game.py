from typing import List

import numpy as np
import torch

from congestion_game.episodic_agent import EpisodicAgent
from congestion_game.utils import one_hot


class EpisodicCongestionGame:
    def __init__(self, agents: List[EpisodicAgent], num_actions, g_func, u_func, history_len, episode_len):
        self.agents = agents
        self.N = len(agents)
        self.A = num_actions
        self.g_func = g_func
        self.u_func = u_func
        self.H = history_len
        self.T = episode_len

    def reset(self):
        for agent in self.agents:
            agent.state = agent.init_state
            agent.history = []
            agent.start_new_episode()

    def step(self):
        # curr_states = [agent.state[0] for agent in self.agents]
        aug_states = [agent.get_augmented_state(self.H) for agent in self.agents]
        actions_and_logprobs = [agent.act(s) for agent, s in zip(self.agents, aug_states)]
        actions = torch.stack([elem[0] for elem in actions_and_logprobs])
        g_term = self.g_func(actions, self.N)
        u_terms = [self.u_func(self.agents[i].state[0], actions[i]) for i in range(self.N)]
        rewards = [g_term + u for u in u_terms]
        potential = g_term + sum(u_terms)
        logprobs = torch.stack([elem[1] for elem in actions_and_logprobs], dim=0)

        self.update_states(actions)
        return actions, logprobs, rewards, potential

    def do_episode(self):
        episode_actions = []
        episode_logprobs = []
        episode_rewards = []
        episode_potentials = []
        for _ in range(self.T):
            actions, logprobs, rewards, potential = self.step()
            episode_actions.append(actions)
            episode_logprobs.append(logprobs)
            episode_rewards.append(rewards)
            episode_potentials.append(potential)

        return episode_actions, episode_logprobs, episode_rewards, episode_potentials

    def update_states(self, actions):
        for i, agent in enumerate(self.agents):
            new_state = np.array([actions[i]])
            agent.update_state(new_state)

