from typing import List

import numpy as np
import torch

from congestion_game.episodic_agent import EpisodicAgent
from congestion_game.utils import one_hot


class EpisodicCongestionGame:
    def __init__(self, agents: List[EpisodicAgent], num_actions, g_func, u_func, history_len, episode_len, device):
        self.agents = agents
        self.N = len(agents)
        self.A = num_actions
        self.g_func = g_func
        self.u_func = u_func
        self.H = history_len
        self.T = episode_len

        self.device = device

    def reset(self):
        for agent in self.agents:
            agent.state = agent.init_state
            agent.history = []
            agent.start_new_episode()

    def step(self):
        # curr_states = [agent.state[0] for agent in self.agents]
        aug_states = [agent.get_augmented_state() for agent in self.agents]
        actions_and_logprobs = [agent.act(s) for agent, s in zip(self.agents, aug_states)]
        actions = torch.stack([elem[0] for elem in actions_and_logprobs])
        g_term = self.g_func(actions, self.N)
        u_terms = [self.u_func(self.agents[i].state.item(), actions[i]) for i in range(self.N)]
        rewards = [g_term + u for u in u_terms]
        potential = g_term + sum(u_terms)
        logprobs = torch.stack([elem[1] for elem in actions_and_logprobs], dim=0)

        self.update_states(actions)
        return actions, logprobs, rewards, potential
    # def step(self):
    #     # Preallocate augmented state tensor if possible
    #     aug_states = torch.stack([agent.get_augmented_state() for agent in self.agents], dim=0)  # Shape: (N, state_dim)
    #
    #     # Actions and logprobs
    #     actions = torch.empty((self.N,), dtype=torch.long, device=self.device)
    #     logprobs = torch.empty((self.N,), dtype=torch.float32, device=self.device)
    #
    #     for i, agent in enumerate(self.agents):
    #         action, logprob = agent.act(aug_states[i])
    #         actions[i] = action
    #         logprobs[i] = logprob
    #
    #     # Shared/global term
    #     g_term = self.g_func(actions, self.N)
    #
    #     # Local utilities
    #     # Assuming agent.state is a 1D tensor or scalar tensor already on the GPU
    #     states = torch.stack([agent.state for agent in self.agents], dim=0)
    #     u_terms = torch.empty((self.N,), dtype=torch.float32, device=self.device)
    #     for i in range(self.N):
    #         u_terms[i] = self.u_func(states[i], actions[i])
    #
    #     rewards = g_term + u_terms  # vectorized reward: one per agent
    #     potential = g_term + u_terms.sum()
    #
    #     self.update_states(actions)
    #     return actions, logprobs, rewards, potential

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
    # def do_episode(self):
    #     # Example: assuming each value returned from step() is 1D scalar (shape: [])
    #     # If shape is (batch_size,), adjust prealloc accordingly
    #
    #     episode_actions = torch.empty((self.T, self.N), dtype=torch.long, device=self.device)
    #     episode_logprobs = torch.empty((self.T, self.N), dtype=torch.float32, device=self.device)
    #     episode_rewards = torch.empty((self.T, self.N), dtype=torch.float32, device=self.device)
    #     episode_potentials = torch.empty((self.T,), dtype=torch.float32, device=self.device)
    #
    #     for t in range(self.T):
    #         actions, logprobs, rewards, potential = self.step()
    #
    #         # If these are scalar tensors, use item() or indexing. But prefer avoiding `.item()` for GPU tensors
    #         episode_actions[t] = actions
    #         episode_logprobs[t] = logprobs
    #         episode_rewards[t] = rewards
    #         episode_potentials[t] = potential
    #
    #     return episode_actions, episode_logprobs, episode_rewards, episode_potentials

    def update_states(self, actions):
        for i, agent in enumerate(self.agents):
            new_state = actions[i].detach()
            agent.update_state(new_state)

