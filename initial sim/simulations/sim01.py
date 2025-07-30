import numpy as np

import matplotlib

from reward_functions import SimpleTwoAgentsReward
from simulations.sim_utils import MDP, Agent, MultiAgent, MultiAgentSimulation

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt


if __name__ == "__main__":
    # Example usage:

    # Define MDP parameters
    states = {0, 1}
    actions = {0, 1}
    transition_prob1 = {
        0: {0: [1., 0.], 1: [0., 1.]},
        1: {0: [1., 0.], 1: [0., 1.]}
    }

    transition_prob2 = {
        0: {0: [0., 1.], 1: [0., 1.]},
        1: {0: [1., 0.], 1: [1., 0.]}
    }



    start_state = 0

    # Initialize MDPs for each agent
    mdp1 = MDP(states, actions, transition_prob1, start_state)
    mdp2 = MDP(states, actions, transition_prob2, start_state)

    # Initialize agents
    agent1 = Agent(id=1, mdp=mdp1)
    agent2 = Agent(id=2, mdp=mdp2)
    multi_agent = MultiAgent(agents=[agent1, agent2], multi_agent_reward=SimpleTwoAgentsReward())

    # Run Value Iteration - joint
    multi_agent.joint_value_iteration()

    # Run Value Iteration - decoupled
    # multi_agent.decoupled_value_iteration()

    # Run simulation
    n_steps = 10
    simulation = MultiAgentSimulation(multi_agent=multi_agent, max_steps=n_steps)
    simulation.run_simulation()

    # Print results
    n_agents = len(simulation.history)
    for step in range(n_steps):
        print(f"Step {step:}")
        print(f" \t Reward: {simulation.joint_reward_history[step]}")
        for i, agent_hist_dict in enumerate(simulation.history):
            print(f" \t Agent {i}: State = {agent_hist_dict['states'][step]}, Action = {agent_hist_dict['actions'][step]}")

    welfare = simulation.calc_accumulated_reward()

    plt.figure()
    plt.plot(np.arange(n_steps), welfare)
    plt.show()