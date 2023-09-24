"""This file contains test of stationary posterior models.
"""
import numpy as np
import matplotlib.pyplot as plt

from envs import InsuranceMarketCt
from stationary_models import StationaryLogisticModel
from scipy.special import expit


# Initialize the environment
env = InsuranceMarketCt(2, 0.15, 0.5, 1., 0.15*np.sqrt(0.5+0.5*2))

# Initialize number of simulations
no_sim = 1
# Initialize number of time steps
T = 20

# Initialize actions
action_set = np.linspace(0.3, 0.9, 5)
no_actions = action_set.shape[0]

# Initialize action frequencies
agent1_frequencies = np.zeros(shape=(T, no_actions))
agent2_frequencies = np.zeros(shape=(T, no_actions))

# Initialize dimesion and dummy action
dimension = 2
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

model_stationary = StationaryLogisticModel(dimension=dimension)
model_stationary_iterative = StationaryLogisticModel(dimension=dimension)
model_laplace = StationaryLogisticModel(dimension=dimension)

# Initialize function to get action vector


def get_action_vector(action_index):
    """Get action vector from action index."""
    action_vector = np.ones(shape=(dimension, 1))
    for i in range(1, dimension):
        action_vector[i, 0] = action_vector[i-1, 0]*action_set[action_index]
    return action_vector


# Perform posterior updates
for it in range(50):
    xk = get_action_vector(action_indices[it])
    mean, cov = model_stationary.update(xk, observations[it])
    mean_2, cov_2 = model_stationary_iterative.iterative_update(
        xk, observations[it], iterations=3)
    mean_3, cov_3 = model_laplace.laplace_update(xk, observations[it])

    print(f'Update observation no {it}')
    print(f'Stationary', mean, cov)
    print(f'Iterative stationary', mean_2, cov_2)
    print(f'Laplace stationary', mean_3, cov_3)


# Showing computed parameters
actions = np.linspace(0, 2, 100)
parameters_stationary = model_stationary.mean.flatten()
mux_stationary = 0
for i in range(dimension):
    mux_stationary += parameters_stationary[i]*np.power(actions, i)
probabilities_stationary = expit(mux_stationary)
parameters_iterative = model_stationary_iterative.mean.flatten()
mux_iterative = 0
for i in range(dimension):
    mux_iterative += parameters_iterative[i]*np.power(actions, i)
probabilities_iterative = expit(mux_iterative)
parameters_laplace = model_laplace.mean.flatten()
mux_laplace = 0
for i in range(dimension):
    mux_laplace += parameters_laplace[i]*np.power(actions, i)
probabilities_laplace = expit(mux_laplace)
plt.plot(actions, probabilities_stationary, label='stationary posterior')
plt.plot(actions, probabilities_iterative,
         label='stationary iterative update posterior')
plt.plot(actions, probabilities_laplace,
         label='stationary laplace update posterior')
plt.grid()
plt.legend()
plt.scatter(action_set, frequencies)
plt.xlabel('actions')
plt.ylabel('frequencies')
plt.savefig(f'images/posterior_stationary_model_test.png')
plt.show()
plt.close()
