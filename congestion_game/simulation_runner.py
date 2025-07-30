import numpy as np
import torch
import itertools
from tqdm import tqdm
import os
import pickle as pkl

from congestion_game.episodic_agent import EpisodicAgent
from congestion_game.episodic_congestion_game import EpisodicCongestionGame
from congestion_game.policies import make_linear_softmax_policy, AgentPolicy, DiscreteStatePolicy
from congestion_game.reward_functions import g_func, make_u_i, make_potential_func
import torch.optim as optim

from congestion_game.utils import evaluate_policy, visualize_joint_mdp


def compute_discounted_return(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    """
    Computes the discounted return for a single trajectory.

    Args:
        rewards (torch.Tensor): Tensor of shape [T] with rewards.
        gamma (float): Discount factor.

    Returns:
        torch.Tensor: Scalar tensor with the total discounted return.
    """
    T = rewards.shape[0]
    discounts = gamma ** torch.arange(T, dtype=rewards.dtype, device=rewards.device)
    return torch.sum(rewards * discounts)


def find_joint_optimum(num_agents, num_states, num_actions, joint_reward_func, gamma=0.99, theta: float = 1e-1):
    joint_states = list(itertools.product(range(num_states), repeat=num_agents))
    joint_actions = list(itertools.product(range(num_actions), repeat=num_agents))

    # Compute optimal value and policy
    V = torch.zeros([num_states] * num_agents)
    pbar = tqdm(desc="Value Iteration")
    policy = {}  # optimal action per joint state
    # num_iterations = 500
    # for _ in tqdm(range(num_iterations)):
    while True:
        delta = 0
        V_new = V.clone()
        for s in joint_states:
            best_val = float('-inf')
            best_action = None
            for a in joint_actions:
                r = joint_reward_func(torch.tensor(s), torch.tensor(a))
                next_state = a  # deterministic transition
                val = r + gamma * V[next_state]
                if val > best_val:
                    best_val = val
                    best_action = a
            V_new[s] = best_val
            policy[s] = best_action
            delta = max(delta, abs(V_new[s] - V[s]))
        V = V_new
        pbar.update(1)
        pbar.set_postfix(delta=f"{delta:.2f}")
        if delta < num_agents * theta:
            break
    pbar.close()
    # Print results
    print("Optimal policy:")
    for s in joint_states:
        print(f"State {s} → Action {policy[s]}")
    return policy


def run_simulation(env, steps):
    all_actions, all_rewards = [], []
    for _ in range(steps):
        actions, rewards = env.step()
        all_actions.append(actions)
        all_rewards.append(rewards)
    return np.stack(all_actions), np.stack(all_rewards)


def train_with_advantage(env: EpisodicCongestionGame, num_episodes, batch_size: int = 1, debug: bool = True):
    # Set optimizers
    independent_optimizers = []
    for i, agent in enumerate(env.agents):
        independent_optimizers.append(optim.SGD(agent.policy_func.parameters(), lr=1e-3))

    agents_losses = [[] for _ in env.agents]
    agents_returns = [[] for _ in env.agents]
    episode_potential_sums = []

    for episode in tqdm(range(num_episodes // batch_size), desc="REINFORCE Batching"):
        # Collect logprobs and rewards for batch
        all_agent_logprobs = [[] for _ in env.agents]
        all_agent_returns = [[] for _ in env.agents]
        all_episode_potentials = []

        for b in range(batch_size):
            env.reset()
            actions, logprobs, rewards, potentials = env.do_episode()

            for i, agent in enumerate(env.agents):
                # agent_rewards = rewards[:, i]
                agent_rewards = torch.stack([step_rewards[i] for step_rewards in rewards])
                agent_logprobs = torch.stack([logprob for action, logprob in agent.policy_map.values()])
                # agent.policy_map[:, i]

                ret = compute_discounted_return(agent_rewards.detach(), gamma=gamma)

                all_agent_logprobs[i].append(agent_logprobs)
                all_agent_returns[i].append(ret)

            all_episode_potentials.append(compute_discounted_return(torch.stack(potentials), gamma=gamma))

        # Compute mean return (baseline) per agent
        agent_baselines = [torch.mean(torch.stack(returns)) for returns in all_agent_returns]

        agents_episode_losses = []
        agents_episode_returns = []
        for i, agent in enumerate(env.agents):
            # Subtract baseline and compute REINFORCE loss
            loss = 0.0
            for logprobs, R in zip(all_agent_logprobs[i], all_agent_returns[i]):
                advantage = R - agent_baselines[i]
                loss += -torch.sum(logprobs) * advantage
            loss = loss / batch_size
            agents_episode_losses.append(loss)
            agents_episode_returns.append(torch.mean(torch.stack(all_agent_returns[i])).cpu().item())

        # Backprop and update
        for i, (optimizer, loss) in enumerate(zip(independent_optimizers, agents_episode_losses)):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            agents_losses[i].append(loss.item())
            agents_returns[i].append(agents_episode_returns[i])

        episode_potential_sums.append(torch.mean(torch.stack(all_episode_potentials)).cpu().item())
        if debug:
            # all_agents_flat_grads = []
            print(f"############################")
            print(f"Episode {episode} Gradeints:")
            for i, agent in enumerate(env.agents):
                agent_grad_norms = []
                print(f"Agent {i} Gradeints:")
                for name, param in agent.policy_func.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        print(f"{name}: grad norm = {grad_norm:.4f}")
                        agent_grad_norms.append(grad_norm)
                # flat_grad_norms = torch.cat(agent_grad_norms)
                # all_agents_flat_grads.append(flat_grads)

                # import matplotlib.pyplot as plt
                #
                # plt.figure()
                # plt.hist(agent_grad_norms, bins=50, alpha=0.5)
                # plt.title(f"Agent {i} - episode {episode} - Gradient Histogram")
                # plt.xlabel("Gradient value")
                # plt.ylabel("Frequency")
                # plt.show()
                print("")
                print("#############################")
                print("")

    last_episode_discounted_potentials = [(gamma ** t) * p.cpu().item() for t, p in enumerate(potentials)]
    return agents_losses, episode_potential_sums, last_episode_discounted_potentials, actions, agents_returns


def train(env: EpisodicCongestionGame, num_episodes, batch_size: int = 1, debug: bool = True):
    # set optimizers
    independent_optimizers = []
    for i, agent in enumerate(env.agents):
        independent_optimizers.append(optim.SGD(agent.policy_func.parameters(), lr=1e-3))

    # iterate episodes
    agents_losses = [[] for _ in env.agents]
    episode_potential_sums = []
    for episode in tqdm(range(num_episodes // batch_size)):
        agents_episode_losses = [0.0 for _ in env.agents]
        for b in range(batch_size):
            env.reset()
            actions, logprobs, rewards, potentials = env.do_episode()

            for i, agent in enumerate(env.agents):
                # Compute loss: use negative log-likelihood weighted by reward
                # loss = -∑ log π(a|s) * reward
                agent_rewards = torch.stack([step_reward[i] for step_reward in rewards])
                agent_logprobs = torch.stack([logprob for action, logprob in agent.policy_map.values()])
                agents_episode_losses[i] += (-torch.sum(agent_logprobs) * compute_discounted_return(agent_rewards.detach(), gamma=gamma)) / batch_size
            episode_potential_sums.append(compute_discounted_return(torch.stack(potentials), gamma=gamma))
        # Before backward pass
        for i, (optimizer, loss) in enumerate(zip(independent_optimizers, agents_episode_losses)):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            agents_losses[i].append(loss.item())

        if debug:
            # all_agents_flat_grads = []
            print(f"############################")
            print(f"Episode {episode} Gradeints:")
            for i, agent in enumerate(env.agents):
                agent_grad_norms = []
                print(f"Agent {i} Gradeints:")
                for name, param in agent.policy_func.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        print(f"{name}: grad norm = {grad_norm:.4f}")
                        agent_grad_norms.append(grad_norm)
                # flat_grad_norms = torch.cat(agent_grad_norms)
                # all_agents_flat_grads.append(flat_grads)

                # import matplotlib.pyplot as plt
                #
                # plt.figure()
                # plt.hist(agent_grad_norms, bins=50, alpha=0.5)
                # plt.title(f"Agent {i} - episode {episode} - Gradient Histogram")
                # plt.xlabel("Gradient value")
                # plt.ylabel("Frequency")
                # plt.show()
                print("")
                print("#############################")
                print("")

    last_episode_discounted_potentials = [(gamma ** t) * p for t, p in enumerate(potentials)]
    return agents_losses, episode_potential_sums, last_episode_discounted_potentials, actions


def run_episodic_simulation(env: EpisodicCongestionGame, num_episodes):
    all_actions, all_rewards = [], []
    for _ in range(num_episodes):
        env.reset()
        actions, rewards = env.do_episode()
        all_actions.append(actions)
        all_rewards.append(rewards)
    return np.stack(all_actions), np.stack(all_rewards)


def get_joint_state_trajectory(initial_state, joint_actions):
    """
    Reconstruct the joint state trajectory based on deterministic transitions.

    Args:
        initial_state (Tuple[int]): Initial joint state.
        joint_actions (List[Tuple[int]]): Sequence of joint actions (one per timestep).

    Returns:
        List[Tuple[int]]: Joint state at each timestep (including initial).
    """
    trajectory = [initial_state]
    for action in joint_actions:
        trajectory.append(action)  # since state = action
    return trajectory


def plot_trajectory(trajectory):
    import networkx as nx

    # Example joint trajectory (each state is a tuple of agent positions)
    G = nx.DiGraph()
    for i in range(len(trajectory) - 1):
        G.add_edge(trajectory[i], trajectory[i+1])

    # Draw
    plt.figure(figsize=(8, 5))
    pos = nx.spring_layout(G, seed=0)
    nx.draw(G, pos, with_labels=True, node_color="lightgreen", node_size=1000, arrowsize=20)
    nx.draw_networkx_labels(G, pos, labels={s: str(s) for s in trajectory})
    plt.title("Trajectory of Joint States")
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('TkAgg')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_agents = 3
    state_dim = 3
    action_dim = state_dim
    history_len = 1
    episode_len = 64
    gamma = 0.99

    BATCH_SIZE = 64
    NUM_EPISODES = BATCH_SIZE * 64


    aug_dim = history_len + 1
    init_states_tuple = torch.tensor([1, 0, 1, 2, 0], device=device)

    overwrite_optimal_policy = False

    agents = []


    for i in range(num_agents):
        init_state = init_states_tuple[i]
        # policy = make_linear_softmax_policy(aug_dim, action_dim)
        policy = DiscreteStatePolicy(state_vocab_sizes=aug_dim * [state_dim],
                                     embedding_dim=max(1, aug_dim // 2) * state_dim,
                                     hidden_dim=action_dim,
                                     num_actions=action_dim)
        policy.to(device)
        agent = EpisodicAgent(state_dim=state_dim,
                              action_dim=action_dim,
                              history_length=history_len,
                              policy_func=policy,
                              init_state=init_state,
                              device=device)
        agents.append(agent)

    ecg = EpisodicCongestionGame(agents=agents,
                                 num_actions=action_dim,
                                 g_func=g_func,
                                 u_func=make_u_i(state_dim),
                                 history_len=history_len,
                                 episode_len=episode_len,
                                 device=device)

    policy_path = f'optimal_policies/{num_agents}_agents_{state_dim}_states_{action_dim}_actions_gamma_{gamma}'
    if os.path.exists(policy_path) or overwrite_optimal_policy:
        with open(policy_path, 'rb') as f:
            optimal_policy = pkl.load(f)
    else:
        optimal_policy = find_joint_optimum(num_agents=num_agents,
                                            num_states=state_dim,
                                            num_actions=action_dim,
                                            joint_reward_func=make_potential_func(state_dim),
                                            gamma=gamma)
        with open(policy_path, 'wb') as f:
            pkl.dump(optimal_policy, f)
    opt_policy_induced_transitions = {(s, a): a for s, a in optimal_policy.items()}
    visualize_joint_mdp(opt_policy_induced_transitions)
    optimal_episode_potential, optimal_episode_step_potentials = evaluate_policy(ecg.agents, optimal_policy, make_potential_func(state_dim), gamma=0.99, episode_len=episode_len)
    agents_losses, potentials, last_episode_potentials, last_actions, returns = train_with_advantage(ecg, NUM_EPISODES, batch_size=BATCH_SIZE, debug=False)
    joint_actions_as_tuples = [tuple(action.tolist()) for action in last_actions]
    traj = get_joint_state_trajectory(init_states_tuple[:num_agents], joint_actions_as_tuples)

    from matplotlib import pyplot as plt

    def moving_average(data, window_size):
        return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


    window_size = min(episode_len // 2, 25)
    smoothed_losses = [moving_average(loss, window_size) for loss in agents_losses]

    plt.figure()
    plt.plot(optimal_episode_step_potentials)
    plt.plot(last_episode_potentials)
    plt.show()

    plt.figure()
    for i in range(num_agents):
        plt.plot(returns[i], label=f'agent {i} - Original')
        # plt.plot(range(window_size - 1, len(agents_losses[i])), smoothed_losses[i], label=f'agent {i} - {window_size}-ep MA', linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Return with Moving Average")
    plt.legend()
    plt.show(block=False)

    plt.figure()
    for i in range(num_agents):
        plt.plot(agents_losses[i], label=f'agent {i} - Original')
        # plt.plot(range(window_size - 1, len(agents_losses[i])), smoothed_losses[i], label=f'agent {i} - {window_size}-ep MA', linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("Loss with Moving Average")
    plt.legend()
    plt.show(block=False)

    plt.figure()
    smoothed_potentials = moving_average(potentials, window_size)
    plt.plot(potentials, label='Original')
    # plt.plot(range(window_size - 1, len(potentials)), smoothed_potentials, label=f'{window_size}-ep MA', linewidth=2)
    plt.axhline(y=optimal_episode_potential, color='red', linestyle='--', label='joint optimum')
    plt.xlabel("Episode")
    plt.ylabel("Potential")
    plt.title("Potential with Moving Average")
    plt.legend()
    plt.show(block=False)

    print(traj)
    plot_trajectory(traj)


    x = 0


