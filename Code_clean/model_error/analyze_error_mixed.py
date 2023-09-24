"""This script analyzes discrepancy between the model and the environment"""

from model_error import optimize_model, environment_functions

from envs import InsuranceMarketCt

import numpy as np
import matplotlib.pyplot as plt

import utils as ut

logger = ut.get_logger(__name__)
logger.setLevel(ut.logging.INFO)


no_actions = 5  # Number of actions
action_set = np.linspace(0.1, 0.9, no_actions, endpoint=True)  # Action set


# Read common parameters from yaml file
params = ut.read_parameters('common_parameters.yaml')
env_params = params['environment_parameters']

# # Change!!!
# env_params['S_c'] = 0.5

# Define logistic model
dimension = 2
logistic_model = optimize_model.LogisticModel(
    action_set=action_set, dimension=dimension)

# Define opponent actions


def sample_reward_probabilities(S_i, env_params, mixed_actions, no_samples=1000):
    """Sample reward probabilities for a mixed action."""
    env = InsuranceMarketCt(**env_params)
    avg_reward = 0
    avg_prob = 0
    actions = np.zeros((no_samples, 2))
    actions[:, 1] = np.random.choice(
        mixed_actions, size=no_samples, replace=True)
    actions[:, 0] = S_i
    rewards, observations = env.step(actions)
    avg_prob = np.mean(observations[:, 0])
    avg_reward = np.mean(rewards[:, 0])
    return avg_prob, avg_reward


plot_range = np.linspace(0, 1, 100)
plotting_model = optimize_model.LogisticModel(
    action_set=plot_range, dimension=dimension)


true_probs = np.zeros_like(action_set)
true_rews = np.zeros_like(action_set)

for i, S_i in enumerate(action_set):
    sample_probs, sample_rewards = \
        sample_reward_probabilities(
            S_i=S_i, env_params=env_params, mixed_actions=[0.1, 0.3], no_samples=5000)

    true_probs[i] = sample_probs
    true_rews[i] = sample_rewards


error_function = optimize_model.get_error_function(
    logistic_model, true_probs)

res = optimize_model.optimize_model(error_function=error_function)
logger.info(f'Optimization result: {res}')
computed_probs = logistic_model.call(res.x)
errors = np.sqrt(((computed_probs - true_probs)**2))


model_params = [1.6139583, -8.12757105]
print('True', true_probs * action_set)
print('Computed', computed_probs * action_set)
print('Game', logistic_model.call(model_params) * action_set)

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(plot_range, plotting_model.call(res.x),
         label='Modeled probability')
plt.plot(plot_range, plotting_model.call(model_params),
         label='From game')
plt.scatter(action_set, true_probs, label='True probability')
plt.grid()
plt.xlabel('S_i')
plt.ylabel('Probability')
plt.legend()
plt.savefig('images/model_error/model_discrepancy_mixed.png')
plt.show()
