import numpy as np
import torch
from torch import nn


class EpisodicAgent:
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 history_length: int,
                 policy_func: nn.Module,
                 init_state=None,
                 device='cpu'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.policy_func = policy_func  # used to sample the per-episode mapping
        self.init_state = torch.tensor([torch.randint(state_dim, size=[1])], device=device) if init_state is None else init_state
        self.state = self.init_state.clone()
        self.history = []
        self.history_len = history_length
        self.policy_map = {}

        self.device = device

    def start_new_episode(self):
        self.policy_map = {}  # clear old mappings

    def act(self, augmented_state):
        state_key = augmented_state.clone()
        if state_key not in self.policy_map:
            probs = self.policy_func(augmented_state)
            self.policy_map[state_key] = probs
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            self.policy_map[state_key] = action, log_prob
        return self.policy_map[state_key]

    def update_state(self, new_state):
        if self.history_len > 0 and len(self.history) == self.history_len:
            self.history.pop(0)

        self.history.append(self.state.clone())
        self.state = new_state

    def get_augmented_state(self):
        if self.history_len == 0:
            return self.state
        else:
            past = self.history if len(self.history) == self.history_len else \
                   [torch.zeros_like(self.state)] * (self.history_len - len(self.history)) + self.history
            return torch.stack(past + [self.state])


def discretize_state(state, bins=10):
    """Simple discretizer for continuous state"""
    return tuple(np.floor(state * bins).astype(int))
