import numpy as np
import torch
from matplotlib import pyplot as plt
import networkx as nx


def one_hot(index, dim):
    vec = np.zeros(dim)
    vec[index] = 1
    return vec


def evaluate_policy(agents, policy, reward_fn, gamma=0.99, episode_len=20):
    total_reward = 0.0
    discount = 1.0

    discounted_rewards = []

    joint_state = tuple(agent.state[0] for agent in agents)
    for t in range(episode_len):
        if joint_state not in policy:
            break  # End of policy coverage or absorbing state

        joint_action = policy[joint_state]
        r = reward_fn(torch.tensor(joint_state), torch.tensor(joint_action))
        total_reward += discount * r
        discounted_rewards.append(discount * r)

        discount *= gamma

        joint_state = tuple([a for a in joint_action])  # deterministic transition

    return total_reward, discounted_rewards


def visualize_joint_mdp(policy_induced_transition_prob):
    plt.figure()
    # Initialize a directed graph
    G = nx.DiGraph()

    # Add edges for joint state-action transitions
    for (joint_state, joint_action), next_state in policy_induced_transition_prob.items():
        # assuming deteremistic
        prob = 1.
        # Add an edge: current joint state -> next joint state
        G.add_edge(joint_state, next_state, action=joint_action, prob=prob)

    # Visualize the graph
    pos = nx.spring_layout(G)  # Layout for positioning nodes
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color="skyblue", font_size=10, font_weight="bold")

    # Draw edge labels (joint action and probability)
    edge_labels = {
        (u, v): f"A: {data['action']}\nP: {data['prob']:.2f}"
        for u, v, data in G.edges(data=True)
    }
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    # Draw directed edges explicitly without ambiguity
    nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=20, connectionstyle="arc3,rad=0.1")

    plt.title("Joint State-Action MDP Visualization")
    plt.show()


def compute_discounted_returns(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    """
    Computes the discounted return for a single trajectory.

    Args:
        rewards (torch.Tensor): Tensor of shape [T] with rewards.
        gamma (float): Discount factor.

    Returns:
        torch.Tensor: Scalar tensor with the total discounted return.
    """
    returns = torch.zeros_like(rewards)
    # R = torch.zeros((rewards.shape[1],), device=rewards.device)
    R = torch.tensor(0)
    for t in reversed(range(rewards.shape[0])):
        R = rewards[t] + gamma * R
        returns[t] = R

    return returns
