"""This script analyzes discrepancy between the model and the environment"""

from model_error import optimize_model, environment_functions

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
opponent_action_set = np.linspace(
    0.1, 1.1, no_actions+1, endpoint=True)  # Action set


plot_range = np.linspace(0, 1, 100)
plotting_model = optimize_model.LogisticModel(
    action_set=plot_range, dimension=dimension)

no_samples = 500
no_opponents = 3

# env_params['N'] = no_opponents+1

opponent_margins = np.random.choice(
    opponent_action_set, size=(no_samples, no_opponents))


errors = np.zeros(shape=(no_actions, no_samples))
sup_errors = np.zeros(shape=(no_samples))

for no_sample in range(no_samples):
    opponent_margins_sample = opponent_margins[no_sample, :].tolist()
    true_probs, true_rewards = \
        environment_functions.expected_reward_probability(
            env_params=env_params,
            S_i=action_set, S=opponent_margins_sample
        )
    error_function = optimize_model.get_error_function(
        logistic_model, true_probs)

    res = optimize_model.optimize_model(error_function=error_function)
    # logger.info(f'Optimization result: {res}')
    computed_probs = logistic_model.call(res.x)
    errors[:, no_sample] = np.sqrt(((computed_probs - true_probs)**2))

    # Compute supremum error
    true_probs_wide, _ = \
        environment_functions.expected_reward_probability(
            env_params=env_params,
            S_i=plot_range, S=opponent_margins_sample
        )
    sup_errors[no_sample] = np.max(
        np.abs(true_probs_wide - plotting_model.call(res.x)))


print(f'Mean error: {np.mean(errors, axis=1)}')

# plt.hist(sup_errors, bins=40)
# plt.xlabel('supremum_errors')
# plt.ylabel('frequency')
# plt.show()

max_errors = np.max(errors, axis=0)
plt.hist(max_errors, bins=100)
plt.xlabel('max_errors')
plt.ylabel('frequency')
plt.show()

print(f'Max error {np.max(errors, axis=1)}')
worst_index = np.argmax(max_errors)

opponent_margins_sample = opponent_margins[worst_index, :].tolist()
print('Worst opponent margins', opponent_margins_sample)
true_probs, true_rewards = \
    environment_functions.expected_reward_probability(
        env_params=env_params,
        S_i=action_set, S=opponent_margins_sample
    )
error_function = optimize_model.get_error_function(
    logistic_model, true_probs)

res = optimize_model.optimize_model(error_function=error_function)


plt.plot(plot_range, plotting_model.call(res.x),
         label='Modeled probability')
plt.plot(plot_range, environment_functions.expected_reward_probability(
    env_params=env_params, S_i=plot_range, S=opponent_margins_sample)[0],
    label='True probability')
plt.xlabel('S_i')
plt.ylabel('Probability')
plt.legend()
plt.grid()
plt.show()
