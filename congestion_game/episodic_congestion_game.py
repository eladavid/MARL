from typing import List, Callable

import numpy as np
import torch
from tqdm import tqdm

from congestion_game.episodic_agent import EpisodicAgent
from congestion_game.utils import one_hot, compute_discounted_returns


class EpisodicCongestionGame:
    def __init__(self,
                 agents: List[EpisodicAgent],
                 num_actions: int,
                 g_func: Callable,
                 u_funcs: List[Callable],
                 history_len: int,
                 episode_len: int,
                 use_episodic_freeze: bool,
                 gamma: float = 0.99,):
        self.agents = agents
        self.N = len(agents)
        self.A = num_actions
        self.g_func = g_func
        self.u_funcs = u_funcs
        self.H = history_len
        self.T = episode_len

        self.use_episodic_freeze = use_episodic_freeze
        self.gamma = gamma

    def reset(self):
        for agent in self.agents:
            agent.state = agent.init_state
            agent.history = []
            agent.start_new_episode()

    def step(self, is_inference: bool = False):
        # curr_states = [agent.state[0] for agent in self.agents]
        aug_states = [agent.get_augmented_state(self.H) for agent in self.agents]
        if is_inference:
            actions_and_logprobs = [agent.argmax_inference(s)
                                    for agent, s in zip(self.agents, aug_states)]
        else:
            actions_and_logprobs = [agent.act(s, use_episodic_freeze=self.use_episodic_freeze)
                                    for agent, s in zip(self.agents, aug_states)]
        actions = torch.stack([elem[0] for elem in actions_and_logprobs])
        g_term = self.g_func(actions, self.N)
        u_terms = [self.u_funcs[i](self.agents[i].state[0], actions[i]) for i in range(self.N)]
        rewards = [g_term + u for u in u_terms]
        potential = g_term + sum(u_terms)
        logprobs = torch.stack([elem[1] for elem in actions_and_logprobs], dim=0)

        self.update_states(actions)
        return actions, logprobs, rewards, potential

    def do_episode(self, is_inference: bool = False):
        episode_actions = []
        episode_logprobs = []
        episode_rewards = []
        episode_potentials = []
        for _ in range(self.T):
            actions, logprobs, rewards, potential = self.step(is_inference=is_inference)
            episode_actions.append(actions)
            episode_logprobs.append(logprobs)
            episode_rewards.append(rewards)
            episode_potentials.append(potential)

        return episode_actions, episode_logprobs, episode_rewards, episode_potentials

    def update_states(self, actions):
        for i, agent in enumerate(self.agents):
            new_state = np.array([actions[i]])
            agent.update_state(new_state)

    def get_all_sampling_functions(self):
        all_agents_sampling_functions = [agent.all_possible_policy_maps(self.H) for agent in self.agents]
        return all_agents_sampling_functions

    def check_if_nash_eq(self):
        is_nash_eq = True
        self.reset()
        all_policy_maps = self.get_all_sampling_functions()
        agents_argmax_policy_maps = []
        for other_agent in self.agents:
            agents_argmax_policy_maps.append(other_agent.get_argmax_policy_map(self.H))
            other_agent.policy_map = agents_argmax_policy_maps[-1]
        # calc argmax return per agent
        argmax_returns = []
        episode_actions, episode_logprobs, episode_rewards, episode_potentials = self.do_episode(is_inference=False)
        for i in range(self.N):
            agent_rewards = torch.stack([step_reward[i] for step_reward in episode_rewards])
            returns = compute_discounted_returns(agent_rewards.detach(), gamma=self.gamma)
            argmax_returns.append(returns[0])

        for i, agent in enumerate(self.agents):
            all_agent_policy_maps = all_policy_maps[i]
            # agent i ran over all policy maps. find best discounted return for agent i. determine if best is current policy
            for ii, agent_i_policy in tqdm(enumerate(all_agent_policy_maps), desc=f"agent {i} policy comparison"):
                self.reset()
                # other agents frozen to argmax policy
                for j, other_agent in enumerate(self.agents):
                    if j != i:
                        other_agent.policy_map = agents_argmax_policy_maps[j]

                # agent i gets the fixed policy
                agent.policy_map = agent_i_policy

                episode_actions, episode_logprobs, episode_rewards, episode_potentials = self.do_episode(is_inference=False)
                agent_rewards = torch.stack([step_reward[i] for step_reward in episode_rewards])
                returns = compute_discounted_returns(agent_rewards.detach(), gamma=self.gamma)

                # if found a policy that strictly beats the argmax
                if returns[0] > argmax_returns[i]:
                    return False
        return True






