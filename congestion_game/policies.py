import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


def make_linear_softmax_policy(input_dim, action_dim):
    """
    Returns a linear-softmax policy function:
    π(s) = softmax(W @ s), where W ∈ ℝ^{A x D}
    """
    W = np.random.randn(action_dim, input_dim) * 0.1

    def policy_fn(state_aug):
        logits = W @ state_aug
        logits -= np.max(logits)  # for numerical stability
        exps = np.exp(logits)
        return exps / np.sum(exps)

    policy_fn.params = W  # for updates (e.g., in REINFORCE)
    return policy_fn


class AgentPolicy(nn.Module):
    def __init__(self, input_dim, num_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, num_actions),
            # nn.ReLU(),
            # nn.Linear(64, num_actions),
        )

    def forward(self, state_tensor):
        logits = self.net(state_tensor)
        return Categorical(logits=logits)


class DiscreteStatePolicy(nn.Module):
    def __init__(self, state_vocab_sizes, embedding_dim, hidden_dim, num_actions):
        """
        state_vocab_sizes: list of vocabulary sizes for each position in the state vector.
        embedding_dim: size of embedding for each discrete element.
        hidden_dim: size of hidden layer.
        num_actions: number of actions (output dimension).
        """
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, embedding_dim) for vocab_size in state_vocab_sizes
        ])
        self.fc = nn.Sequential(
            nn.Linear(len(state_vocab_sizes) * embedding_dim, num_actions),
            # nn.ReLU(),
            # nn.Linear(hidden_dim, num_actions)
        )

    def forward(self, state_vector):
        # state_vector: Tensor of shape (batch_size, state_dim)
        embedded = [
            emb(state_vector[i]) for i, emb in enumerate(self.embeddings)
        ]  # List of (batch_size, embedding_dim)
        x = torch.cat(embedded, dim=-1)  # (batch_size, state_dim * embedding_dim)
        logits = self.fc(x)
        probs = torch.softmax(logits, dim=-1)
        return probs

class DiscreteStatePolicyNoEmbeddings(nn.Module):
    def __init__(self, state_vocab_sizes, hidden_dim, num_actions):
        super().__init__()
        self.state_vocab_sizes = state_vocab_sizes
        self.total_input_dim = sum(state_vocab_sizes)

        self.fc = nn.Sequential(
            nn.Linear(self.total_input_dim, num_actions)
        )

        # Initialize once (you can remove this line if you want purely random)
        # nn.init.constant_(self.fc[0].weight, 0.0)
        # nn.init.constant_(self.fc[0].bias, 0.0)
        self.reset_parameters()

    def forward(self, state_vector):
        one_hots = [
            torch.nn.functional.one_hot(state_vector[i], num_classes=vocab_size).float()
            for i, vocab_size in enumerate(self.state_vocab_sizes)
        ]
        x = torch.cat(one_hots, dim=-1)
        logits = self.fc(x)
        probs = torch.softmax(logits, dim=-1)
        return probs

    def reset_parameters(self):
        """Reinitialize the policy with fresh random weights."""
        layer = self.fc[0]  # Linear layer

        # Xavier uniform is a good default for logits
        nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(layer.bias)

    def load_parameters(self, params):
        """
        Load parameters from a state-dict or from another policy.

        Args:
            params: either:
                - a PyTorch state_dict (dict of tensors), or
                - another instance of DiscreteStatePolicyNoEmbeddings.
        """
        if isinstance(params, nn.Module):
            params = params.state_dict()

        # Strict=True ensures exact shape match.
        self.load_state_dict(params, strict=True)


def project_onto_simplex(P: torch.Tensor) -> torch.Tensor:
    """
    Project each row of P onto the probability simplex:
        { x >= 0, sum(x) = 1 }.
    P: tensor of shape (N, A)
    Returns: tensor of same shape with each row on the simplex.
    """
    # Sort in descending order along last dimension
    sorted_P, _ = torch.sort(P, dim=-1, descending=True)  # (N, A)
    cumsum = torch.cumsum(sorted_P, dim=-1)                # (N, A)
    A = P.size(-1)

    # j = 1..A for each row
    js = torch.arange(1, A + 1, device=P.device, dtype=P.dtype).view(1, -1)

    # Condition: sorted_P_j + (1 - cumsum_j) / j > 0
    cond = sorted_P + (1 - cumsum) / js > 0

    # rho = max j satisfying condition (0-based index)
    rho = cond.sum(dim=-1) - 1                             # (N,)

    # θ = (sum_{j<=rho} sorted_P_j - 1) / (rho + 1)
    idx = rho.view(-1, 1)
    theta = (cumsum.gather(1, idx) - 1) / (idx.to(P.dtype) + 1)  # (N,1)

    # Project
    P_proj = P - theta
    P_proj = torch.clamp(P_proj, min=0.0)

    # Renormalize for numerical stability
    P_proj = P_proj / P_proj.sum(dim=-1, keepdim=True)
    return P_proj


class DirectTabularPolicy(nn.Module):
    """
    Direct parametrization: parameters ARE the probabilities π(a | s).

    state_vocab_sizes: list of vocab sizes for each discrete state component.
                       e.g. [N1, N2, ..., Nk].
    Internally we store probs_table with shape:
        (N1, N2, ..., Nk, num_actions),
    and index it with the discrete state_vector.
    """
    def __init__(self, state_vocab_sizes, num_actions, exploration_rate: float = 0.1):
        super().__init__()
        self.state_vocab_sizes = state_vocab_sizes
        self.num_actions = num_actions
        self.exploration_rate = exploration_rate

        # Total number of discrete states
        total_states = int(np.prod(state_vocab_sizes))

        # Sample independent Dirichlet(num_actions) random probability vectors
        flat = torch.distributions.Dirichlet(torch.ones(num_actions)).sample((total_states,))
        probs = flat.view(*state_vocab_sizes, num_actions)
        # This parameter is directly π(s,a), not logits.
        self.probs_table = nn.Parameter(probs, requires_grad=True)

    def reset_parameters(self):
        """Reinitialize the policy with fresh random weights."""
        total_states = int(np.prod(self.state_vocab_sizes))
        flat = torch.distributions.Dirichlet(torch.ones(self.num_actions)).sample((total_states,))
        probs = flat.view(*self.state_vocab_sizes, self.num_actions)
        # This parameter is directly π(s,a), not logits.
        self.probs_table = nn.Parameter(probs, requires_grad=True)

    def forward(self, state_vector):
        """
        state_vector: discrete state indices.
        Supports:
          - shape (batch_size, state_dim), or
          - shape (state_dim, batch_size)  (to match your existing policies).
        Returns:
          probs of shape (batch_size, num_actions).
        """
        # Normalize to (batch_size, state_dim)
        if state_vector.dim() == 1:
            # single state, shape (state_dim,)
            batch = 1
            sv = state_vector.unsqueeze(0)  # (1, state_dim)
        elif state_vector.dim() == 2:
            # Either (batch, state_dim) or (state_dim, batch)
            if state_vector.size(0) == len(self.state_vocab_sizes):
                # assume (state_dim, batch)
                sv = state_vector.transpose(0, 1)  # -> (batch, state_dim)
            else:
                sv = state_vector  # already (batch, state_dim)
            batch = sv.size(0)
        else:
            raise ValueError("state_vector must be 1D or 2D tensor")

        # Build index tuple per dimension: each is (batch,)
        idx_per_dim = []
        for i, vocab_size in enumerate(self.state_vocab_sizes):
            idx_i = sv[:, i].long()
            if idx_i.min() < 0 or idx_i.max() >= vocab_size:
                raise ValueError(f"State index out of range for dim {i}: "
                                 f"min={idx_i.min().item()}, max={idx_i.max().item()}, "
                                 f"vocab_size={vocab_size}")
            idx_per_dim.append(idx_i)

        # Fancy indexing into probs_table: shape (batch, num_actions)
        idx = tuple(idx_per_dim)
        probs = self.probs_table[idx]  # (batch, num_actions)

        # Because we project after updates, probs should already be on simplex.
        # Optionally clamp tiny negatives due to numerical error.
        probs = torch.clamp(probs, min=0.0)
        # Optional renorm for extra safety
        probs = probs / probs.sum(dim=-1, keepdim=True)
        probs = (1 - self.exploration_rate) * probs + self.exploration_rate * (torch.ones_like(probs.detach()) / self.num_actions)
        return probs.squeeze()

    @torch.no_grad()
    def project_parameters_onto_simplex(self):
        """
        Project every π(s, :) row onto the simplex.
        Call this after optimizer.step().
        """
        flat = self.probs_table.view(-1, self.num_actions)   # (num_states_total, A)
        flat_proj = project_onto_simplex(flat)               # project row-wise
        self.probs_table.copy_(flat_proj.view_as(self.probs_table))



import copy

class PolicyArchive:
    def __init__(self, max_size):
        self.max_size = max_size
        self.saved_profiles = []  # list of list[state_dict]

    def add_profile(self, agents):
        # snapshot of all agent policies
        profile = [copy.deepcopy(agent.policy_func.state_dict()) for agent in agents]

        # append new profile
        self.saved_profiles.append(profile)

        # enforce size limit: keep only last max_size
        if len(self.saved_profiles) > self.max_size:
            self.saved_profiles.pop(0)  # remove oldest

    def get_profile(self, idx):
        return self.saved_profiles[idx]

    def num_profiles(self):
        return len(self.saved_profiles)
