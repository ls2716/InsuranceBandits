"""This file contains simulation of game between bandits
on the insurance environment.
"""
import numpy as np
import matplotlib.pyplot as plt

from envs import InsuranceMarket
from bandits import EpsGreedy, EXP3, UCBTuned, UCBV
from ARC_model_based import ARCPricingModel
from scipy.special import expit


import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 13})


np.random.seed(0)

# Initialize the environment
env = InsuranceMarket(2, 0.15, 0.5, 1., 0.15*np.sqrt(0.5+0.5*2))

# Initialize number of simulations
no_sim = 1
# Initialize number of time steps
T = 20

# Initialize actions
action_set = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
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
no_samples = 200
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


eta = 0.002
# Perform posterior updates
for it in range(len(observations)):
    # print(f'Action index {action_indices[it]} observation {observations[it]}')
    m, d = agent1.update(action_indices[it], observations[it])
    m2, d2 = agent2.update2(action_indices[it], observations[it])
    # print('Update 1')
    # print(m, d)
    # print('Update 2')
    # print(m2, d2)
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
plt.figure(figsize=(7, 5))
plt.plot(actions, probabilities1, label='logistic model learned')
# plt.plot(actions, probabilities2,
#          label='Ad-hoc method (no sigma/variance scaling)')
plt.grid()
plt.scatter(action_set, frequencies, label='frequencies from data')
plt.legend()
plt.xlabel('margin $S_i$')
plt.ylabel('frequency of offer fill $P(Y=1)$')
plt.title('logistic model learned')
plt.savefig(f'posterior_test_poster.png', dpi=600)
# plt.show()
plt.close()

exit(0)

# # Showing eta updates
# actions = np.linspace(0, 2, 100)

# etas = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.]
# for eta in etas:
#     print(f'Eta: {eta}')
#     agent1.reset()
#     # Perform posterior updates
#     for it in range(len(observations)):
#         # print(f'Action index {action_indices[it]} observation {observations[it]}')
#         m, d = agent1.update3(action_indices[it], observations[it], eta=eta)
#         agent1.m = m
#         agent1.d = d

#     parameters = agent1.m.flatten()
#     mux = 0
#     for i in range(dimension):
#         mux += parameters[i]*np.power(actions, i)
#     probabilities = expit(mux)
#     plt.plot(actions, probabilities,
#              label=f'$\eta = {eta}$')

# plt.grid()
# plt.legend()
# plt.scatter(action_set, frequencies)
# plt.xlabel('actions')
# plt.ylabel('frequencies')
# plt.title(f'No samples = {no_samples}')
# plt.savefig(f'posterior_test_etas_no_{no_samples}.png')
# plt.show()


# Showing eta updates
actions = np.linspace(0, 2, 100)

etas = [1., 2., 2.3, 2.5, 2.7, 3., 4., 5.]
for eta in etas:
    print(f'Eta: {eta}')
    agent1.reset()
    # Perform posterior updates
    for it in range(len(observations)):
        # print(f'Action index {action_indices[it]} observation {observations[it]}')
        m, d = agent1.update4(action_indices[it], observations[it], eta=eta)
        agent1.m = m
        agent1.d = d

    parameters = agent1.m.flatten()
    mux = 0
    for i in range(dimension):
        mux += parameters[i]*np.power(actions, i)
    probabilities = expit(mux)
    plt.plot(actions, probabilities,
             label=f'$\eta = {eta}$')

plt.grid()
plt.legend()
plt.scatter(action_set, frequencies)
plt.xlabel('actions')
plt.ylabel('frequencies')
plt.title(f'No samples = {no_samples}')
plt.savefig(f'posterior_test_paper_etas_no_{no_samples}.png')
plt.show()
