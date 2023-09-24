"""This file contains simulation of game between bandits
on the insurance environment.
"""
import numpy as np
import matplotlib.pyplot as plt

from envs import InsuranceMarket
from bandits import EpsGreedy, EXP3, UCBTuned, UCBV
from TS_mb_all_obs import TSPricingModel
from tqdm import tqdm
import os


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
        # if (i+1) % 1000 == 0:
        #     print('Current t', i)
        #     print('Current m', agent1.m)
        #     print('Current d', agent1.d)

    return reward_history, action1_history, action2_history


# Initialize the environment
env = InsuranceMarket(2, 0.15, 0.5, 1., 0.15*np.sqrt(0.5+0.5*2))

# Initialize number of simulations
no_sim = 20
# Initialize number of time steps
T = 1000

# Initialize actions
action_set = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
# action_set = np.linspace(0.1, 1., 4)
no_actions = action_set.shape[0]


def agent_name(agent):
    agent_name = str(agent).split('.')[1].split(' ')[0]
    return agent_name


# Setup meta game - each agent vs each agent
agent_epsgreedy = EpsGreedy(eps=0.01, candidate_margins=action_set)
agent_EXP3 = EXP3(gamma=0.01, candidate_margins=action_set)
agent_UCBV = UCBV(candidate_margins=action_set)
agent_UCBTuned = UCBTuned(candidate_margins=action_set)
agent_TS = TSPricingModel(candidate_margins=action_set,
                          rho=0.005, beta=0.99)

# Pairs to try
agent_pairs = [
    (agent_TS, agent_epsgreedy),
    # (agent_TS, agent_EXP3),
    # (agent_TS, agent_UCBV),
    # (agent_TS, agent_UCBTuned),
    # (agent_TS, agent_TS),
]

foldername = 'meta_game_TSmb_images_10_actions'
# Create folder
try:
    os.mkdir(foldername)
except:
    print("Directory could not be created - probably already exists")


for agent_pair in agent_pairs:
    agent1, agent2 = agent_pair

    agent1_name = agent_name(agent1)
    agent2_name = agent_name(agent2)

    print(f"Working on {agent1_name} vs. {agent2_name}")
    print(f"Number of simulations: {no_sim}")
    print(f"Timesteps: {T}")

    # Initialize action frequencies
    agent1_frequencies = np.zeros(shape=(T, no_actions))
    agent2_frequencies = np.zeros(shape=(T, no_actions))
    mean_rewards = np.zeros(shape=(T, 2))

    for sim in tqdm(range(no_sim)):
        agent1.reset()
        agent2.reset()
        reward_history, action1_history, action2_history = single_game(
            env, agent1, agent2, T)
        agent1_frequencies += action1_history/no_sim
        agent2_frequencies += action2_history/no_sim

        print('Mean reward', np.mean(reward_history, axis=0))
        mean_rewards = reward_history/no_sim

    fig, axs = plt.subplots(3, 1, figsize=(6, 10))
    for i in range(no_actions):
        axs[0].plot(agent1_frequencies[:, i], label=f's={action_set[i]}')
    axs[0].set_title(f'{agent1_name} action frequencies')
    axs[0].set_xlabel('time')
    axs[0].set_ylabel('freqiency')
    axs[0].legend()
    axs[0].grid()

    for i in range(no_actions):
        axs[1].plot(agent2_frequencies[:, i], label=f's={action_set[i]}')
    axs[1].set_title(f'{agent2_name} action frequencies')
    axs[1].set_xlabel('time')
    axs[1].set_ylabel('freqiency')
    axs[1].legend()
    axs[1].grid()

    # Plot running average of the rewards
    window = 50
    for i in range(window, T):
        mean_rewards[i, :] = np.mean(reward_history[i-window:i, :], axis=0)
    axs[2].plot(mean_rewards[:, 0], label=f'{agent1_name}')
    axs[2].plot(mean_rewards[:, 1], label=f'{agent2_name}')
    axs[2].set_title(f'Mean rewards')
    axs[2].set_xlabel('time')
    axs[2].set_ylabel('reward')
    axs[2].legend()
    axs[2].grid()

    plt.savefig(
        f'{foldername}/game_sim{no_sim}_T{T}_a1{agent1_name}_agent2{agent2_name}.png')
    plt.close()
