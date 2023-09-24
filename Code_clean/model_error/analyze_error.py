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


fig, axs = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Model discrepancy for different opponent actions')

errors = np.zeros((no_actions+1, no_actions))
supremum_errors = np.zeros(no_actions+1)

for i, opponent_action in enumerate(opponent_action_set):
    true_probs, true_rewards = \
        environment_functions.expected_reward_probability(
            env_params=env_params,
            S_i=action_set, S=[opponent_action]
        )
    error_function = optimize_model.get_error_function(
        logistic_model, true_probs)

    res = optimize_model.optimize_model(error_function=error_function)
    logger.info(f'Optimization result: {res}')
    computed_probs = logistic_model.call(res.x)
    errors[i, :] = np.sqrt(((computed_probs - true_probs)**2))

    # Compute supremum error
    true_probs_wide, _ = \
        environment_functions.expected_reward_probability(
            env_params=env_params,
            S_i=plot_range, S=[opponent_action]
        )
    supremum_errors[i] = np.max(
        np.abs(true_probs_wide - plotting_model.call(res.x)))

    plot_i, plot_j = i//3, i % 3
    axs[plot_i, plot_j].plot(plot_range, plotting_model.call(res.x),
                             label='Modeled probability')
    axs[plot_i, plot_j].plot(plot_range, environment_functions.expected_reward_probability(
        env_params=env_params, S_i=plot_range, S=[opponent_action])[0],
        label='True probability')
    axs[plot_i, plot_j].scatter(action_set, true_probs,
                                label='Optimisation points')
    axs[plot_i, plot_j].grid()
    axs[plot_i, plot_j].legend()
    axs[plot_i, plot_j].set_xlabel('S_i')
    axs[plot_i, plot_j].set_ylabel('Probability')
    axs[plot_i, plot_j].set_title(
        f'Opponent action: {opponent_action:.2f}')
plt.tight_layout()
plt.savefig('images/model_error/model_discrepancy_N_2.png')
# plt.show()

print("Supremum errors:")
print(supremum_errors)


logger.info(f'Errors:\n {errors}')

# Print errors in a latex format
print('Errors:')
print('Opponent action & ' +
      ' & '.join([f'{action:.1f}' for action in action_set]))
print('\\\\')
print('\\hline')
for i, error in enumerate(errors):
    print(f'{opponent_action_set[i]:.1f} & ' +
          ' & '.join([f'{error_value:.1e}' for error_value in error]) + f' & {supremum_errors[i]:.1e}', end=" \\\\\n")
