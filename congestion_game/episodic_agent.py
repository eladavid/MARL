import itertools

import numpy as np
import torch
import copy

class EpisodicAgent:
    def __init__(self, state_dim, action_dim, policy_func, init_state=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.policy_func = policy_func  # used to sample the per-episode mapping
        self.init_state = np.array([np.random.randint(state_dim)]) if init_state is None else np.array([init_state])
        self.state = np.copy(self.init_state)
        self.history = []
        self.policy_map = {}

    def get_params(self):
        # Deep-copy state_dict so later operations cannot modify it
        return copy.deepcopy(self.policy_func.state_dict())

    def set_params(self, state_dict):
        # Restore parameters
        self.policy_func.load_state_dict(copy.deepcopy(state_dict))

    def start_new_episode(self):
        self.policy_map = {}  # clear old mappings

    def act(self, augmented_state, use_episodic_freeze: bool = False):
        state_key = tuple(augmented_state)
        if state_key not in self.policy_map:
            probs = self.policy_func(torch.tensor(augmented_state, dtype=torch.long))
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

    def get_argmax_policy_map(self, history_len):
        # iterate over all states and produce the argmax policy map
        state_vocab_sizes = [self.state_dim for _ in range(history_len + 1)]

        # All possible augmented states (Cartesian product of vocabularies)
        all_augmented_states = list(itertools.product(*[range(v) for v in state_vocab_sizes]))

        argmax_policy_map = {}
        for augmented_state in all_augmented_states:
            action, log_prob = self.argmax_inference(augmented_state)
            argmax_policy_map[tuple(augmented_state)] = action.detach(), log_prob.detach()
        return argmax_policy_map

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

    import itertools

    def all_possible_policy_maps(self, history_len):
        """
        Generate all possible deterministic policy maps from augmented states to actions.

        Args:
            state_vocab_sizes (List[int]): For each position in the augmented state vector, its vocabulary size.
            num_actions (int): Number of discrete actions.

        Returns:
            List[Dict[Tuple[int, ...], int]]: A list of policy maps.
                Each map is a dict: augmented_state_tuple -> action
        """
        state_vocab_sizes = [self.state_dim for _ in range(history_len + 1)]
        num_actions = self.action_dim

        # All possible augmented states (Cartesian product of vocabularies)
        all_augmented_states = list(itertools.product(*[range(v) for v in state_vocab_sizes]))

        # All possible policy choices: for each state, assign an action
        # Generate all functions: product of actions repeated |states| times
        all_action_assignments = itertools.product(range(num_actions), repeat=len(all_augmented_states))

        # Build policy maps
        all_policy_maps = []
        for assignment in all_action_assignments:
            policy_map = {
                state: (torch.tensor(action), torch.tensor(0.)) for state, action in zip(all_augmented_states, assignment)
            }
            all_policy_maps.append(policy_map)

        return all_policy_maps

