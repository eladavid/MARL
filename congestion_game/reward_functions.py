import torch


def g_func(actions, num_agents):
    """
    actions: Tensor of shape (n_agents,) with discrete actions
    action_dim: total number of possible actions (needed to pad bincount)
    """
    counts = torch.bincount(actions)  # (A,)
    return 5*(1-torch.sum((counts.float() / num_agents) ** 2))

def make_u_i(num_states):
    def u(s_i, a_i):
        return 1*(1-(abs(s_i - a_i) / num_states))
    return u

def make_potential_func(state_dim):
    def potential_func(joint_state, joint_action):
        num_agents = len(joint_state)
        g_term = g_func(joint_action, num_agents)
        u_func = make_u_i(state_dim)
        u_sum = sum([u_func(s_i, a_i) for s_i, a_i in zip(joint_state, joint_action)])
        return g_term + u_sum
    return potential_func


def make_random_g_func(num_agents, num_actions, seed=None):
    if seed is not None:
        torch.manual_seed(seed)

    shape = [num_actions] * num_agents
    reward_table = torch.rand(*shape)  # Random values for each joint action

    def g_func(actions, num_agents):
        # actions: Tensor of shape (num_agents,)
        return reward_table[tuple(actions.tolist())]

    return g_func

def make_random_u_funcs(num_agents, num_states, num_actions, seed=None):
    if seed is not None:
        torch.manual_seed(seed)

    u_funcs = []
    for i in range(num_agents):
        u_table = torch.rand(num_states, num_actions)
        # Shuffle values to diversify behavior across agents
        u_table = u_table[torch.randperm(num_states)]
        def u_func_factory(u_table_copy):
            def u_func(s_i, a_i):
                return u_table_copy[s_i, a_i]
            return u_func
        u_funcs.append(u_func_factory(u_table))

    return u_funcs