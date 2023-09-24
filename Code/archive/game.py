"""This file contains simulation of game between bandits
on the insurance environment.
"""
import numpy as np
import matplotlib.pyplot as plt

from envs import InsuranceMarket
from bandits import EpsGreedy, EXP3, UCBTuned, UCBV
from ARC import ARC_Pricing


def single_game(env, agent1, agent2, T):
    """Simulate game between bandints for T episodes"""
    no_actions = agent1.actions.shape[0]
    reward_history = np.zeros(shape=(T, 2))
    action1_history = np.zeros(shape=(T, no_actions))
    action2_history = np.zeros(shape=(T, no_actions))
    for i in range(T):
        action1, action1_index = agent1.get_action()
        action2, action2_index = agent2.get_action()
        rewards = env.step(np.array([action1, action2])).flatten()
        agent1.update(action1_index, rewards[0])
        agent2.update(action2_index, rewards[1])
        reward_history[i, :] = rewards[:]
        action1_history[i, action1_index] = 1.
        action2_history[i, action2_index] = 1.

    return reward_history, action1_history, action2_history


# Initialize the environment
env = InsuranceMarket(2, 0.2, 0, 1., 0.2)

# Initialize number of simulations
no_sim = 500
# Initialize number of time steps
T = 500

# Initialize actions
action_set = np.array([0., 0.25, 0.5, 0.75, 1., 1.25])
no_actions = action_set.shape[0]

# Initialize action frequencies
agent1_frequencies = np.zeros(shape=(T, no_actions))
agent2_frequencies = np.zeros(shape=(T, no_actions))

# Initialize agents
agent1 = EpsGreedy(eps=0.01, candidate_margins=action_set)
agent2 = EpsGreedy(eps=0.01, candidate_margins=action_set)
agent2 = ARC_Pricing(candidate_margins=action_set, rho=0.1, beta=0.99, sigma=5)

for sim in range(no_sim):
    agent1.reset()
    agent2.reset()
    reward_history, action1_history, action2_history = single_game(
        env, agent1, agent2, T)
    agent1_frequencies += action1_history/no_sim
    agent2_frequencies += action2_history/no_sim


fig, axs = plt.subplots(2, 1, figsize=(6, 10))
for i in range(no_actions):
    axs[0].plot(agent1_frequencies[:, i], label=f's={action_set[i]}')
axs[0].set_title('Agent1 action frequencies')
axs[0].set_xlabel('time')
axs[0].set_ylabel('freqiency')
axs[0].legend()
axs[0].grid()

for i in range(no_actions):
    axs[1].plot(agent2_frequencies[:, i], label=f's={action_set[i]}')
axs[1].set_title('Agent2 action frequencies')
axs[1].set_xlabel('time')
axs[1].set_ylabel('freqiency')
axs[1].legend()
axs[1].grid()

plt.savefig('test_game.png')
plt.close()
