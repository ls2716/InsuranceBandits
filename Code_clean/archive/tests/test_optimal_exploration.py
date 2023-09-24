"""This file contains test between naive and optimal exploration.
"""
import numpy as np
import matplotlib.pyplot as plt

from envs import InsuranceMarketCt
from stationary_models import StationaryLogisticModel
from scipy.special import expit


# Initialize the environment
env = InsuranceMarketCt(2, 0.15, 0.5, 1., 0.15*np.sqrt(0.5+0.5*2))


# Initialize actions
action_set = np.linspace(0.3, 0.9, 5)
no_actions = action_set.shape[0]

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


model_one_by_one = StationaryLogisticModel(dimension=dimension)
model_optimal_cov_decrease = StationaryLogisticModel(dimension=dimension)

number_of_samples = 80

# Initialize function to get action vector


def get_action_vector(action_index):
    """Get action vector from action index."""
    action_vector = np.ones(shape=(dimension, 1))
    for i in range(1, dimension):
        action_vector[i, 0] = action_vector[i-1, 0]*action_set[action_index]
    return action_vector


one_by_one_action = 0

# Define a function which computes the optimal action minimizes the next sigma norm


def get_optimal_action(model, action_set):
    best_action = 0
    best_norm = 100000
    for action in range(no_actions):
        xk = get_action_vector(action)
        norm = model.get_next_norm(xk)
        if norm < best_norm:
            best_norm = norm
            best_action = action
    return best_action


number_of_samples = 100


# Perform posterior updates
for it in range(number_of_samples):
    rewards_step, observations_step = env.step(
        np.array([action_set[one_by_one_action], dummy_action]))
    observation_naive = observations_step[0, 0]
    xk_naive = get_action_vector(one_by_one_action)
    one_by_one_action = (one_by_one_action + 1) % no_actions

    best_action = get_optimal_action(model_optimal_cov_decrease, action_set)
    rewards_step, observations_step = env.step(
        np.array([action_set[best_action], dummy_action]))
    observation_optimal = observations_step[0, 0]
    xk_optimal = get_action_vector(best_action)

    mean, cov = model_one_by_one.laplace_update(xk_naive, observation_naive)
    mean_2, cov_2 = model_optimal_cov_decrease.laplace_update(
        xk_optimal, observation_optimal)

    print(f'Update observation no {it}')
print(f'One by one', mean, cov)
print(f'Optimal', mean_2, cov_2)


# Showing computed parameters
actions = np.linspace(0, 2, 100)
parameters_one_by_one = model_one_by_one.mean.flatten()
mux_one_by_one = 0
for i in range(dimension):
    mux_one_by_one += parameters_one_by_one[i]*np.power(actions, i)
probabilities_one_by_one = expit(mux_one_by_one)
parameters_optimal_cov_decrease = model_optimal_cov_decrease.mean.flatten()
mux_optimal_cov_decrease = 0
for i in range(dimension):
    mux_optimal_cov_decrease += parameters_optimal_cov_decrease[i]*np.power(
        actions, i)
probabilities_optimal_cov_decrease = expit(mux_optimal_cov_decrease)
plt.plot(actions, probabilities_one_by_one, label='naive exploitation')
plt.plot(actions, probabilities_optimal_cov_decrease,
         label='optimal exploitation')
plt.grid()
plt.legend()
plt.scatter(action_set, frequencies)
plt.xlabel('actions')
plt.ylabel('frequencies')
plt.savefig(f'images/optimal_exploration_test.png')
plt.show()
plt.close()
