"""This file contains simulation of game between bandits
on the insurance environment.
"""
import numpy as np
import matplotlib.pyplot as plt

from envs import InsuranceMarket
from bandits import EpsGreedy, EXP3, UCBTuned, UCBV
from ARC_GP import ARCPricingModel
from ARC import ARCPricing
from scipy.special import expit

# np.random.seed(0)

# Initialize the environment
env = InsuranceMarket(2, 0.15, 0.5, 1., 0.15*np.sqrt(0.5+0.5*2))

# Initialize number of simulations
no_sim = 1
# Initialize number of time steps
T = 20

# Initialize actions
action_set = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
no_actions = action_set.shape[0]

# Initialize action frequencies
agent1_frequencies = np.zeros(shape=(T, no_actions))
agent2_frequencies = np.zeros(shape=(T, no_actions))

# Initialize agents
dimension = 2
agent_cb = ARCPricing(candidate_margins=action_set,
                      rho=0.1, beta=0.98, sigma=3)
agent_gp = ARCPricingModel(candidate_margins=action_set,
                           rho=0.1, beta=0.98, sigma=3)
dummy_action = 0.75

# Check reward profile
rewards = np.zeros_like(action_set)
frequencies = np.zeros_like(action_set)
no_samples = 200
action_indices = []
observations = []
for j in range(no_samples):
    for i in range(action_set.shape[0]):
        rewards_step, observations_step = env.step(
            np.array([action_set[i], dummy_action]))
        frequencies[i] += observations_step[0, 0]/no_samples
        rewards[i] += rewards_step[0, 0]/no_samples
        observations.append(rewards_step[0, 0])
        action_indices.append(i)

print('Rewards per action', rewards)
print('Acceptance frequencies', frequencies)


# Perform posterior updates
for it in range(0, len(observations)//20):
    # print(f'Action index {action_indices[it]} observation {observations[it]}')
    m, d = agent_cb.update(action_indices[it], observations[it])
    m2, d2 = agent_gp.update(action_indices[it], observations[it])
    # print('Update')
    # print(m, d)
    agent_cb.m = m
    agent_cb.d = d
    agent_gp.m = m2
    agent_gp.d = d2


# Showing computed parameters
rewards_cb = agent_cb.m.flatten()
parameters_gp = agent_gp.m.flatten()
rewards_gp = [parameters_gp[0]]
for i in range(1, len(action_set)):
    rewards_gp.append(rewards_gp[-1]+parameters_gp[i])
plt.plot(action_set, rewards_cb, label='ARC independent arms')
plt.plot(action_set, rewards_gp, label='ARC GP')
plt.grid()
plt.legend()
plt.scatter(action_set, rewards)
plt.xlabel('actions')
plt.ylabel('rewards')
plt.savefig(f'posterior_test_ARC_GP.png')
plt.show()
plt.close()
