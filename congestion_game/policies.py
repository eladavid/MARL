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