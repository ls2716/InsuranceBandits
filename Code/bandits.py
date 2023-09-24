"""This file contains the implementation of bandit algortihms
for the market problem.
"""
import numpy as np
import random
# Implement eps-greedy bandit


class EpsGreedy(object):
    """eps-greed bandit algorithm as in OTC paper
    """

    def __init__(self, eps, candidate_margins) -> None:
        """Initialize the algorithm.

        Arguments:
            - eps (float): epsilon
            - candidate_margins (numpy array of floats): list of possible margins
        """
        self.eps = eps
        self.actions = candidate_margins
        self.n = np.zeros_like(candidate_margins)
        self.r = np.zeros_like(candidate_margins)
        self.n_actions = candidate_margins.shape[0]

    def reset(self):
        """Reset agent"""
        self.r *= 0
        self.n *= 0

    def get_action(self):
        """Get action.
        """
        # Pick action
        if np.random.random() > self.eps:
            # Best if greedy
            max_r = np.max(self.r)
            max_actions = [i
                           for i in range(self.n_actions) if self.r[i] > max_r-0.001]
            action_index = random.choice(max_actions)
        else:
            # Random uniformly if exploratory
            action_index = np.random.randint(self.n_actions)
        return self.actions[action_index], action_index

    def get_action_probabilities(self):
        """Get action probabilities.
        """
        greedy_action_index = np.argmax(self.r)
        probabilities = np.zeros_like(self.r)
        probabilities[greedy_action_index] = 1-self.eps
        probabilities += self.eps/self.n_actions
        return probabilities

    def update(self, action_index, reward):
        """Update state of rewards.
        """
        self.r[action_index] = (
            self.n[action_index] * self.r[action_index] + reward)/(self.n[action_index]+1)
        self.n[action_index] += 1


class EXP3(object):
    """EXP3 bandit algorithm as in OTC paper
    """

    def __init__(self, gamma, candidate_margins) -> None:
        """Initialize the algorithm

        Arguments:
            - gamma (float): gamma
            - candidate_margins (numpy array of floats): list of possible margins
        """
        self.gamma = gamma
        self.actions = candidate_margins
        self.weights = np.ones_like(candidate_margins)
        self.K = self.actions.shape[0]
        self.n_actions = candidate_margins.shape[0]

    def reset(self):
        """Reset agent"""
        self.weights = self.weights * 0+1

    def get_action(self):
        """Get actions according to batch size
        """
        self.probabilities = (1-self.gamma)*self.weights / \
            np.sum(self.weights) + self.gamma/self.K
        action_index = np.random.choice(self.K, p=self.probabilities)
        return self.actions[action_index], action_index

    def get_action_probabilities(self):
        """Get action probabilities.
        """
        return self.probabilities

    def update(self, action_index, reward):
        """Update state of rewards.
        """
        probability = (1-self.gamma)*self.weights[action_index] / \
            np.sum(self.weights) + self.gamma/self.K
        self.weights[action_index] = self.weights[action_index] * \
            np.exp(self.gamma*reward/(self.K*probability))


class UCBV(object):
    """UCB-V bandit algorithm as in OTC paper
    """

    def __init__(self, candidate_margins) -> None:
        """Initialize the algorithm

        Arguments:
            - candidate_margins (numpy array of floats): list of possible margins
        """
        self.actions = candidate_margins
        self.K = candidate_margins.shape[0]
        self.n_actions = candidate_margins.shape[0]
        self.r = np.zeros_like(candidate_margins)
        self.r2 = np.zeros_like(self.r)
        self.n = np.zeros_like(self.r)
        self.t = 1

    def reset(self):
        """Reset agent"""
        self.r *= 0
        self.r2 *= 0
        self.n *= 0

    def objective_function(self):
        """Compute the objective function.
        """
        if (np.prod(self.n) == 0):
            return -self.n
        tmp = np.log(self.t)/self.n
        vt = self.r2-self.r**2+1e-9
        if (vt < 0).any():
            print(vt)  # error here
        return self.r + 3*tmp + np.sqrt(2*tmp*vt)

    def get_action(self):
        """Get actions according to batch size
        """
        r = self.objective_function()
        max_r = np.max(r)
        max_actions = [i for i in range(self.n_actions) if r[i] > max_r-0.001]
        action_index = random.choice(max_actions)
        return self.actions[action_index], action_index

    def update(self, action_index, reward):
        """Update state of rewards.
        """
        self.t += 1
        self.r[action_index] = (
            self.n[action_index] * self.r[action_index] + reward)/(self.n[action_index]+1)
        self.r2[action_index] = (
            self.n[action_index] * self.r2[action_index] + reward**2)/(self.n[action_index]+1)
        self.n[action_index] += 1


class UCBTuned(object):
    """UCB-Tuned bandit algorithm as in OTC paper
    """

    def __init__(self, candidate_margins) -> None:
        """Initialize the algorithm

        Arguments:
            - candidate_margins (numpy array of floats): list of possible margins
        """
        self.actions = candidate_margins
        self.K = candidate_margins.shape[0]
        self.n_actions = candidate_margins.shape[0]
        self.r = np.zeros_like(candidate_margins)
        self.r2 = np.zeros_like(self.r)
        self.n = np.zeros_like(self.r)
        self.t = 1

    def reset(self):
        """Reset agent"""
        self.r *= 0
        self.r2 *= 0
        self.n *= 0

    def objective_function(self):
        """Compute the objective function.
        """
        if (np.prod(self.n) == 0):
            return -self.n
        tmp = np.log(self.t)/self.n
        vt = self.r2-self.r**2 + np.sqrt(2*tmp)
        return self.r + np.sqrt(tmp*np.minimum(
            1/4, vt
        ))

    def get_action(self):
        """Get actions according to batch size
        """
        r = self.objective_function()
        max_r = np.max(r)
        max_actions = [i for i in range(self.n_actions) if r[i] > max_r-0.001]
        action_index = random.choice(max_actions)
        return self.actions[action_index], action_index

    def update(self, action_index, reward):
        """Update state of rewards.
        """
        self.t += 1
        self.r[action_index] = (
            self.n[action_index] * self.r[action_index] + reward)/(self.n[action_index]+1)
        self.r2[action_index] = (
            self.n[action_index] * self.r2[action_index] + reward**2)/(self.n[action_index]+1)
        self.n[action_index] += 1


class ARC(object):
    """ARC bandit algorithm as in Nash's paper
    """

    def __init__(self, candidate_margins) -> None:
        """Initialize the algorithm

        Arguments:
            - candidate_margins (numpy array of floats): list of possible margins
        """

    def get_action(self):
        """Get actions according to batch size
        """
        pass

    def update(self, action_index, reward):
        """Update state of rewards.
        """
        pass
