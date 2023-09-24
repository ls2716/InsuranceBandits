"""This file contains simulation of game between bandits
on the insurance environment.
"""
import numpy as np
import matplotlib.pyplot as plt

from envs import InsuranceMarket
from bandits import EpsGreedy, EXP3, UCBTuned, UCBV
from TS_mb_all_obs import TSPricingModel
from scipy.special import expit


import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 13})

np.random.seed(0)


class DummyAgent(object):

    def __init__(self, action) -> None:
        self.action = action

    def get_action(self):
        return self.action, 0

    def update(self, *args):
        pass

    def reset(self):
        pass


def single_game(env, agent1, agent2, T):
    """Simulate game between bandints for T episodes"""
    no_actions = agent1.actions.shape[0]
    reward_history = np.zeros(shape=(T, 2))
    action1_history = np.zeros(shape=(T, no_actions))
    action2_history = np.zeros(shape=(T, no_actions))
    for i in range(T):
        action1, action1_index = agent1.get_action()
        action2, action2_index = agent2.get_action()
        rewards, observations = env.step(
            np.array([action1, action2]))
        rewards = rewards.flatten()
        observations = observations.flatten()
        new_m, new_d = agent1.update_all_observations(
            action1_index, observations[0])
        agent1.m = new_m
        agent1.d = new_d
        agent2.update(action2_index, rewards[1])
        reward_history[i, :] = rewards[:]
        action1_history[i, action1_index] = 1.
        action2_history[i, action2_index] = 1.
        agent1.new_step()
        if (i+1) % 500 == 0:
            print('Current t', i)
            print('Action', action1_index)
            print('Current m', agent1.m)
            print('Current d', agent1.d)

    return reward_history, action1_history, action2_history


# Initialize the environment
env = InsuranceMarket(2, 0.15, 0.5, 1., 0.15*np.sqrt(0.5+0.5*2))

# Initialize number of simulations
no_sim = 30
# Initialize number of time steps
T = 2000

# Initialize actions
action_set = np.linspace(0.1, 1.1, 20)
no_actions = action_set.shape[0]

# Initialize action frequencies
agent1_frequencies = np.zeros(shape=(T, no_actions))
agent2_frequencies = np.zeros(shape=(T, no_actions))

# Initialize agents
agent1 = TSPricingModel(candidate_margins=action_set,
                        rho=0.001, beta=0.98, dimension=2)
dummy_action = 0.3
agent2 = DummyAgent(action=dummy_action)

# # Check reward profile
# rewards = np.zeros_like(action_set)
# frequencies = np.zeros_like(action_set)
# no_samples = 1000
# for i in range(action_set.shape[0]):
#     for j in range(no_samples):
#         rewards_step, observations = env.step(
#             np.array([action_set[i], dummy_action]))
#         frequencies[i] += observations[0, 0]/no_samples
#         rewards[i] += rewards_step[0, 0]/no_samples

# print('Rewards per action', rewards)
# print('Acceptance frequencies', frequencies)

# # Trying to find optimal parameters for logit
# actions = np.linspace(0, 2, 100)
# parameters = [19.8, -25., 0]
# probabilities = expit(
#     parameters[0] + parameters[1]*actions + parameters[2]*actions*actions)
# plt.plot(actions, probabilities)
# plt.scatter(action_set, frequencies)
# plt.xlabel('actions')
# plt.ylabel('probabilities')
# plt.savefig('mb_posterior.png')
# plt.show()

# exit(0)

# Perform simulations and update frequencies of actions
for sim in range(no_sim):
    print('Simulation number ', sim, end='\n')
    agent1.reset()
    agent2.reset()
    reward_history, action1_history, action2_history = single_game(
        env, agent1, agent2, T)
    agent1_frequencies += action1_history/no_sim
    agent2_frequencies += action2_history/no_sim


# plot results
plt.figure(figsize=(7, 5))
for i in range(no_actions):
    plt.semilogx(list(range(1, T+1)),
                 agent1_frequencies[:, i], label=f'$s={action_set[i]}$')
plt.title(f'TS action frequencies with a fixed opponent $s_j=0.3$')
plt.xlabel('time step')
plt.ylabel('action frequency')
plt.legend()
plt.grid()

plt.savefig('test_game_fixed_TS_mb_all_obs_opp0.3.png', dpi=600)
plt.close()
