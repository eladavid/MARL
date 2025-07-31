import numpy as np
import torch


class EpisodicAgent:
    def __init__(self, state_dim, action_dim, policy_func, init_state=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.policy_func = policy_func  # used to sample the per-episode mapping
        self.init_state = np.array([np.random.randint(state_dim)]) if init_state is None else np.array([init_state])
        self.state = np.copy(self.init_state)
        self.history = []
        self.policy_map = {}

    def start_new_episode(self):
        self.policy_map = {}  # clear old mappings

    def act(self, augmented_state, use_episodic_freeze: bool = False):
        state_key = tuple(augmented_state)
        if state_key not in self.policy_map:
            probs = self.policy_func(torch.tensor(augmented_state))
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            if use_episodic_freeze:
                self.policy_map[state_key] = action, log_prob
        else:
            action, log_prob = self.policy_map[state_key]
        return action, log_prob

    def argmax_inference(self, augmented_state):
        probs = self.policy_func(torch.tensor(augmented_state))
        dist = torch.distributions.Categorical(probs)
        action = torch.argmax(probs)
        log_prob = dist.log_prob(action)
        return action, log_prob

    def update_state(self, new_state):
        self.history.append(self.state.copy())
        self.state = new_state

    def get_augmented_state(self, history_len):
        if history_len == 0:
            return self.state
        else:
            past = self.history[-history_len:] if len(self.history) >= history_len else \
                   [np.zeros_like(self.state)] * (history_len - len(self.history)) + self.history
            return np.concatenate(past + [self.state])