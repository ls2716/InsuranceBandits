"""This file contains test of stationary posterior models.
"""
import numpy as np
import matplotlib.pyplot as plt

from envs import InsuranceMarketCt
from stationary_models import StationaryLogisticModel
from scipy.special import expit

from bandits_st_mb import UCBMB

# Initialize the environment
env = InsuranceMarketCt(2, 0.15, 0.5, 1., 0.15*np.sqrt(0.5+0.5*2))

# Initialize number of simulations
no_sim = 1
# Initialize number of time steps
T = 20

# Initialize actions
action_set = np.linspace(0.1, 0.9, 5)
no_actions = action_set.shape[0]

# Initialize dimesion and dummy action
dimension = 2

model_stationary = StationaryLogisticModel(dimension=dimension)
agent = UCBMB(candidate_margins=action_set, model=model_stationary,
              no_samples=3, percentile=0.68)


# # Test sampling function
# model_stationary.mean = np.array([8, -10]).reshape(-1, 1)
# model_stationary.cov = np.eye(2)*5.
action_vectors = agent.action_vectors
# rewards = model_stationary.sample_rewards(
#     action_vectors=action_vectors, no_samples=20)


# print(np.percentile(rewards, 70, axis=0))

# Initialize  dummy action
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


# Initialize function to get action vector
def get_action_vector(action_index):
    """Get action vector from action index."""
    action_vector = np.ones(shape=(dimension, 1))
    for i in range(1, dimension):
        action_vector[i, 0] = action_vector[i-1, 0]*action_set[action_index]
    return action_vector


def action_values(sample_rewards, gamma=0.9):
    """Compute action values from sampled rewards."""
    mean_rewards = np.mean(sample_rewards, axis=0)
    max_mean_reward = np.max(mean_rewards)
    positive_reward = np.maximum(sample_rewards - max_mean_reward, 0)
    negative_reward = np.minimum(sample_rewards - max_mean_reward, 0)
    action_values = np.mean(positive_reward, axis=0) / \
        (1-gamma) + np.mean(negative_reward, axis=0)
    return action_values


# Perform posterior updates
for it in range(200):
    xk = get_action_vector(action_indices[it])
    mean, cov = model_stationary.update(xk, observations[it])

    if it % 20 == 0:
        print(f'Update observation no {it}')
        print(f'Stationary', mean.T, '\n', cov)
        reward_samples = model_stationary.sample_rewards(
            action_vectors=action_vectors, no_samples=100)
        print(f'68% percentile', np.percentile(reward_samples, 68, axis=0))
        print(f'Action values', action_values(reward_samples))
