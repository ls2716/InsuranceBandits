"""This file tests the epsilon greedy bandit algorithm against
a dummy agent which always quotes the same dummy action."""

# Import packages
import numpy as np
import matplotlib.pyplot as plt


# Import bandit algorithms
from bandits_OTC import UCBV

# Import dummy agent
from bandits_OTC import DummyAgent

# Initialize the environment
from envs import InsuranceMarketCt

from stationary_models import StationaryLogisticModel

# Define a single game function between the bandit and the dummy agent


def single_game(bandit: UCBV, dummy_agent: DummyAgent, env: InsuranceMarketCt, T: int):
    no_actions = bandit.n_actions
    reward_history = np.zeros(shape=(T, 2))
    action_bandit_history = np.zeros(shape=(T, no_actions))
    action_dummy_history = np.zeros(shape=(T, no_actions))
    for i in range(T):
        action_bandit, action_bandit_index = bandit.get_action()
        action_dummy, action_dummy_index = dummy_agent.get_action()
        rewards, observations = env.step(
            np.array([action_bandit, action_dummy]))
        rewards = rewards.flatten()
        observations = observations.flatten()
        bandit.update(action_bandit_index, rewards[0])
        reward_history[i, :] = rewards[:]
        action_bandit_history[i, action_bandit_index] = 1.
        action_dummy_history[i, action_dummy_index] = 1.

    return reward_history, action_bandit_history, action_dummy_history


# Initialize the environment
env = InsuranceMarketCt(2, 0.15, 0.5, 1., 0.15*np.sqrt(0.5+0.5*2))

# Initialize number of simulations
no_sim = 100

# Initialize number of time steps
T = 1000

# Initialize actions
no_actions = 5
action_set = np.linspace(0.1, 0.9, no_actions, endpoint=True)
print('Action set', action_set)
no_actions = action_set.shape[0]

# Initialize action frequencies
agent_bandit_frequencies = np.zeros(shape=(T, no_actions))
agent_dummy_frequencies = np.zeros(shape=(T, no_actions))

# Initialize dimesion and dummy action
dimension = 2
dummy_action = 0.75

# Initialize reward history
reward_history = np.zeros(shape=(T, 2))

# Initialize model
model_stationary = StationaryLogisticModel(dimension=dimension)
# Initialize bandit
bandit = UCBV(candidate_margins=action_set)
# Initialize dummy agent
dummy_agent = DummyAgent(action=dummy_action)


# Run simulations
for i in range(no_sim):
    print('Simulation', i+1)
    # Run a single game
    reward_history_step, action_bandit_history_step, action_dummy_history_step = single_game(
        bandit, dummy_agent, env, T)
    # Update reward history
    reward_history += reward_history_step/no_sim
    # Update action frequencies
    agent_bandit_frequencies += action_bandit_history_step/no_sim
    agent_dummy_frequencies += action_dummy_history_step/no_sim

    # print('Mean rewards', bandit.mean_rewards)
    # print('Action values', bandit.action_values)
    # print(bandit.model.mean.T)
    # print(bandit.model.cov)

    # Reset the bandit
    bandit.reset()

# Plot action frequencies of the bandit agent
plt.figure()
for i in range(no_actions):
    plt.plot(agent_bandit_frequencies[:, i], label=f'Action {action_set[i]}')
plt.legend()
plt.xlabel('Time step')
plt.ylabel('Frequency')
plt.title(
    f'Action frequencies of the UCB-V agent vs a dummy agent with action {dummy_action}')
plt.grid()
plt.savefig(
    f'images/test_ucbv_vs_dummy_{dummy_action}_no_actions_{no_actions}.png')
plt.show()


# Plot reward history
plt.figure()
plt.plot(reward_history[:, 0], label='Bandit agent')
plt.title(
    f'Reward history of the UCB-V agent vs a dummy agent with action {dummy_action}')
plt.xlabel('Time step')
plt.ylabel('Reward')
plt.grid()
plt.savefig(
    f'images/test_ucbv_vs_dummy_{dummy_action}_reward_history.png')
