"""This file contains simulation of game between bandits
on the insurance environment.
"""
import numpy as np
import matplotlib.pyplot as plt

from envs import InsuranceMarket
from bandits import EpsGreedy, EXP3, UCBTuned, UCBV
from ARC_model_based import ARCPricingModel
from scipy.special import expit

# np.random.seed(0)

# Initialize the environment
env = InsuranceMarket(2, 0.15, 0.5, 1., 0.15*np.sqrt(0.5+0.5*2))

# Initialize number of simulations
no_sim = 1
# Initialize number of time steps
T = 20

# Initialize actions
action_set = np.linspace(0.3, 0.9, 100)
no_actions = action_set.shape[0]

# Initialize action frequencies
agent1_frequencies = np.zeros(shape=(T, no_actions))
agent2_frequencies = np.zeros(shape=(T, no_actions))

# Initialize agents
dimension = 2
agent1 = ARCPricingModel(candidate_margins=action_set,
                         rho=0.1, beta=0.98, dimension=dimension)
agent2 = ARCPricingModel(candidate_margins=action_set,
                         rho=0.1, beta=0.98, dimension=dimension)
dummy_action = 0.75

# Check reward profile
rewards = np.zeros_like(action_set)
frequencies = np.zeros_like(action_set)
no_samples = 500
action_indices = []
observations = []
for j in range(no_samples):
    for i in range(action_set.shape[0]):
        rewards_step, observations_step = env.step(
            np.array([action_set[i], dummy_action]))
        frequencies[i] += observations_step[0, 0]/no_samples
        rewards[i] += rewards_step[0, 0]/no_samples
        observations.append(observations_step[0, 0])
        action_indices.append(i)

print('Rewards per action', rewards)
print('Acceptance frequencies', frequencies)


# Perform posterior updates
for it in range(500):
    # print(f'Action index {action_indices[it]} observation {observations[it]}')
    m, d = agent1.update_single_obs(action_indices[it], observations[it])
    m2, d2 = agent2.update_all_observations(
        action_indices[it], observations[it])
    # print('Update 1')
    # print(m, d)
    print(f'Update observation no {it}')
    print(m2, d2)
    agent2.new_step()
    # print(agent2._f())
    agent1.m = m
    agent1.d = d
    agent2.m = m2
    agent2.d = d2


# Showing computed parameters
actions = np.linspace(0, 2, 100)
parameters1 = agent1.m.flatten()
mux1 = 0
for i in range(dimension):
    mux1 += parameters1[i]*np.power(actions, i)
probabilities1 = expit(mux1)
parameters2 = agent2.m.flatten()
mux2 = 0
for i in range(dimension):
    mux2 += parameters2[i]*np.power(actions, i)
probabilities2 = expit(mux2)
plt.plot(actions, probabilities1, label='single observation Kalman filter')
plt.plot(actions, probabilities2, label='posterior with all observations')
plt.grid()
plt.legend()
plt.scatter(action_set, frequencies)
plt.xlabel('actions')
plt.ylabel('frequencies')
plt.savefig(f'posterior_test_model_based_ARC_allvssingle.png')
plt.show()
plt.close()
