"""This file contains simulation of game between bandits
on the insurance environment.
"""
import numpy as np
import matplotlib.pyplot as plt

from envs import InsuranceMarket
from bandits import EpsGreedy, EXP3, UCBTuned, UCBV


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
        rewards = env.step(np.array([action1, action2])).flatten()
        agent1.update(action1_index, rewards[0])
        agent2.update(action2_index, rewards[1])
        reward_history[i, :] = rewards[:]
        action1_history[i, action1_index] = 1.
        action2_history[i, action2_index] = 1.
    # print(action1_history.sum(0))
    # exit(0)

    return reward_history, action1_history, action2_history


# Initialize the environment
env = InsuranceMarket(2, 0.15, 0.5, 1., 0.15*np.sqrt(0.5+0.5*2))

# Initialize number of simulations
no_sim = 100
# Initialize number of time steps
T = 1000

# Initialize actions
action_set = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
no_actions = action_set.shape[0]

# Initialize action frequencies


# Initialize agents
agents = [
    EpsGreedy(eps=0.02, candidate_margins=action_set),
    EXP3(gamma=0.1, candidate_margins=action_set),
    UCBV(action_set),
    UCBTuned(action_set)
]
agent2 = DummyAgent(action=0.75)

agent_frequencies = []
# Perform simulations and update frequencies of actions
for agent1 in agents:
    agent1_frequencies = np.zeros(shape=(T, no_actions))
    # agent2_frequencies = np.zeros(shape=(T, no_actions))
    for sim in range(no_sim):
        agent1.reset()
        agent2.reset()
        reward_history, action1_history, action2_history = single_game(
            env, agent1, agent2, T)
        agent1_frequencies += action1_history/no_sim
        # agent2_frequencies += action2_history/no_sim
    agent_frequencies.append(agent1_frequencies.copy())
    print(agent1)
    for i in range(no_actions):
        plt.semilogx(list(range(1, T+1)),
                     agent1_frequencies[:, i], label=f's={action_set[i]}')
    plt.xlabel('time')
    plt.ylabel('frequency')
    plt.legend()
    agent_name = str(agent1).split(' ')[0][9:]
    plt.title(agent_name)
    plt.savefig(agent_name+'_fixed.png')
    plt.close()


# plot results
fig, axs = plt.subplots(2, 2, figsize=(8, 8))
for i in range(4):
    for j in range(no_actions):
        axs[i//2, i % 2].semilogx(list(range(1, T+1)),
                                  agent_frequencies[i][:, j], label=f's={action_set[j]}')
    axs[i//2, i % 2].set_xlabel('time')
    axs[i//2, i % 2].set_ylabel('freqiency')
    axs[i//2, i % 2].legend()
    axs[i//2, i % 2].grid()


axs[0, 0].set_title(f'Eps-greedy action frequencies\n T={T} no_sim={no_sim}')
axs[0, 1].set_title(f'EXP3 action frequencies\n T={T} no_sim={no_sim}')
axs[1, 0].set_title(f'UCB-V action frequencies\n T={T} no_sim={no_sim}')
axs[1, 1].set_title(f'UCB-Tuned action frequencies\n T={T} no_sim={no_sim}')
plt.savefig('test_game_fixed_4.png')
plt.close()
