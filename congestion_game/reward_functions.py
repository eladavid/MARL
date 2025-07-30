import torch


def g_func(actions, num_agents):
    """
    actions: Tensor of shape (n_agents,) with discrete actions
    action_dim: total number of possible actions (needed to pad bincount)
    """
    counts = torch.bincount(actions)  # (A,)
    return 0.2*(1-torch.sum((counts.float() / num_agents) ** 2))


def make_u_i(num_states):
    def u(s_i, a_i):
        return 5*(1-(abs(s_i - a_i) / num_states))
    return u


def make_potential_func(state_dim):
    def potential_func(joint_state, joint_action):
        num_agents = len(joint_state)
        g_term = g_func(joint_action, num_agents)
        u_func = make_u_i(state_dim)
        u_sum = sum([u_func(s_i, a_i) for s_i, a_i in zip(joint_state, joint_action)])
        return g_term + u_sum
    return potential_func
