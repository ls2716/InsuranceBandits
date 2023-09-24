"""This file implements model-based bandit algorithms."""

import numpy as np
import random

from bandits_OTC import BaseBandit


class EpsGreedyMB(BaseBandit):
    """This class implements model based eps-greedy bandit algorithm
    """
    # Intialize the object

    def __init__(self, eps, candidate_margins, model) -> None:
        """Initialize the algorithm.

        Arguments:
            - eps (float): epsilon
            - candidate_margins (numpy array of floats): list of possible margins
            - model (object): model object
        """
        self.eps = eps
        self.actions = candidate_margins
        self.n_actions = candidate_margins.shape[0]
        self.model = model
        # Assemble matrix with all possible action vectors
        self.action_vectors = np.ones(
            shape=(self.model.dimension, self.n_actions))
        for i in range(self.n_actions):

            self.action_vectors[:, i] = self.get_action_vector(i)[:, 0]

    # Reset the agent
    def reset(self):
        """Reset agent"""
        self.model.reset()

    # Evaluate rewards
    def evaluate_rewards(self):
        """Use the model to evaluate rewards for each action"""
        self.rewards = self.model.evaluate_rewards(self.action_vectors)

    # Get action vector

    def get_action_vector(self, k):
        """Get action vector"""
        xk = np.ones(shape=(self.model.dimension, 1))
        for i in range(1, self.model.dimension):
            xk[i, 0] = xk[i-1, 0]*self.actions[k]
        return xk

    # Get action and corresponding action index

    def get_action(self):
        self.evaluate_rewards()
        if random.random() < self.eps:
            action_index = random.randint(0, self.n_actions-1)
        else:
            action_index = np.argmax(self.rewards)
        return self.actions[action_index], action_index

    def update(self, action_index, reward, observation):
        """Update the model based on the observation and the action index.
        Reward is not important.
        """
        xk = self.get_action_vector(action_index)
        _, _ = self.model.update(xk, observation)


class UCBMB(BaseBandit):
    """This class implements model based eps-greedy bandit algorithm
    """
    # Intialize the object

    def __init__(self, candidate_margins, model, no_samples, gamma=0.9) -> None:
        """Initialize the algorithm.

        Arguments:
            - candidate_margins (numpy array of floats): list of possible margins
            - model (object): model object
        """
        self.actions = candidate_margins
        self.n_actions = candidate_margins.shape[0]
        self.model = model
        # Assemble matrix with all possible action vectors
        self.action_vectors = np.ones(
            shape=(self.model.dimension, self.n_actions))
        for i in range(self.n_actions):
            self.action_vectors[:, i] = self.get_action_vector(i)[:, 0]

        self.no_samples = no_samples
        self.gamma = gamma

    # Reset the agent
    def reset(self):
        """Reset agent"""
        self.model.reset()

    # Evaluate rewards
    def evaluate_action_values(self):
        """Use the model to evaluate value of each action"""
        sample_rewards = self.sample_rewards()
        self.mean_rewards = np.mean(sample_rewards, axis=0)
        max_mean_reward = np.max(self.mean_rewards)
        positive_reward = np.maximum(sample_rewards - max_mean_reward, 0)
        negative_reward = np.minimum(sample_rewards - max_mean_reward, 0)
        self.action_values = np.mean(positive_reward, axis=0) / \
            (1-self.gamma) + np.mean(negative_reward, axis=0)
        return self.action_values

    # Get action vector
    def get_action_vector(self, k):
        """Get action vector"""
        xk = np.ones(shape=(self.model.dimension, 1))
        for i in range(1, self.model.dimension):
            xk[i, 0] = xk[i-1, 0]*self.actions[k]
        return xk

    def sample_rewards(self):
        self.reward_samples = self.model.sample_rewards(
            self.action_vectors, no_samples=self.no_samples)
        return self.reward_samples

    # Get action and corresponding action index
    def get_action(self):
        action_values = self.evaluate_action_values()
        action_index = np.argmax(action_values)
        return self.actions[action_index], action_index

    def update(self, action_index, observation):
        """Update the model based on the observation and the action index.
        Reward is not important.
        """
        xk = self.get_action_vector(action_index)
        _, _ = self.model.update(xk, observation)


class UCBMB_P(BaseBandit):
    """This class implements model based eps-greedy bandit algorithm
    """
    # Intialize the object

    def __init__(self, candidate_margins, model, no_samples, percentile=0.68) -> None:
        """Initialize the algorithm.

        Arguments:
            - candidate_margins (numpy array of floats): list of possible margins
            - model (object): model object
        """
        self.actions = candidate_margins
        self.n_actions = candidate_margins.shape[0]
        self.model = model
        # Assemble matrix with all possible action vectors
        self.action_vectors = np.ones(
            shape=(self.model.dimension, self.n_actions))
        for i in range(self.n_actions):
            self.action_vectors[:, i] = self.get_action_vector(i)[:, 0]

        self.no_samples = no_samples
        self.percentile = percentile
        self.T = 1

    # Reset the agent
    def reset(self):
        """Reset agent"""
        self.model.reset()
        self.T = 1

    # Evaluate rewards
    def evaluate_action_values(self):
        """Use the model to evaluate value of each action"""
        sample_rewards = self.sample_rewards()
        self.action_values = np.percentile(
            sample_rewards, (1 - 1/self.T)*100, axis=0)
        return self.action_values

    # Get action vector
    def get_action_vector(self, k):
        """Get action vector"""
        xk = np.ones(shape=(self.model.dimension, 1))
        for i in range(1, self.model.dimension):
            xk[i, 0] = xk[i-1, 0]*self.actions[k]
        return xk

    def sample_rewards(self):
        self.reward_samples = self.model.sample_rewards(
            self.action_vectors, no_samples=self.no_samples)
        return self.reward_samples

    # Get action and corresponding action index
    def get_action(self):
        action_values = self.evaluate_action_values()
        action_index = np.argmax(action_values)
        return self.actions[action_index], action_index

    def update(self, action_index, observation):
        """Update the model based on the observation and the action index.
        Reward is not important.
        """
        xk = self.get_action_vector(action_index)
        _, _ = self.model.update(xk, observation)
        self.T += 1


class UCBMB_STD(BaseBandit):
    """This class implements model based eps-greedy bandit algorithm
    """
    # Intialize the object

    def __init__(self, candidate_margins, model, no_samples, alpha=1.) -> None:
        """Initialize the algorithm.

        Arguments:
            - candidate_margins (numpy array of floats): list of possible margins
            - model (object): model object
        """
        self.actions = candidate_margins
        self.n_actions = candidate_margins.shape[0]
        self.model = model
        # Assemble matrix with all possible action vectors
        self.action_vectors = np.ones(
            shape=(self.model.dimension, self.n_actions))
        for i in range(self.n_actions):
            self.action_vectors[:, i] = self.get_action_vector(i)[:, 0]

        self.no_samples = no_samples
        self.alpha = alpha
        self.T = 1

    # Reset the agent
    def reset(self):
        """Reset agent"""
        self.model.reset()
        self.T = 1

    # Evaluate rewards
    def evaluate_action_values(self):
        """Use the model to evaluate value of each action"""
        sample_rewards = self.sample_rewards()
        self.stds = np.std(sample_rewards, axis=0)
        self.action_values = np.mean(
            sample_rewards, axis=0) + self.stds*self.alpha * np.log(self.T)
        return self.action_values

    # Get action vector
    def get_action_vector(self, k):
        """Get action vector"""
        xk = np.ones(shape=(self.model.dimension, 1))
        for i in range(1, self.model.dimension):
            xk[i, 0] = xk[i-1, 0]*self.actions[k]
        return xk

    def sample_rewards(self):
        self.reward_samples = self.model.sample_rewards(
            self.action_vectors, no_samples=self.no_samples)
        return self.reward_samples

    # Get action and corresponding action index
    def get_action(self):
        action_values = self.evaluate_action_values()
        action_index = np.argmax(action_values)
        return self.actions[action_index], action_index

    def update(self, action_index, observation):
        """Update the model based on the observation and the action index.
        Reward is not important.
        """
        xk = self.get_action_vector(action_index)
        _, _ = self.model.update(xk, observation)
        self.T += 1
