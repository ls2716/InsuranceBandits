"""This file contains implementation of the logistic regression
models for model-based bandits.
"""
import numpy as np
from scipy.special import expit
import scipy
import torch

def sigmoid(x):
    return expit(x)

class NonStationaryLogisticModel(object):
    """Defines the logistic regression model for bandits
    with iterative update for the mean based on of all the observations."""

    def __init__(self, dimension=2, variance=10**3, eta=0.001) -> None:
        """Initialisation function."""
        self.dimension = dimension
        self.variance = variance
        self.eta=eta
        self.reset()
        self.f_lambda = np.pi/5.35
    
    def reset(self):
        """Reset model."""
        self.mean = np.zeros(shape=(self.dimension,1))
        self.cov = np.eye(self.dimension)*self.variance
        self.cov_0 = np.eye(self.dimension)*self.variance
        self.xks = None
        self.Ys = None
        self.t = 0
    

    def stabilise(self, m_threshold=100, t_stabilisation=0):
        """A stabilistation function for the model."""

        # If the mean is too large, set it to 0
        if np.max(np.abs(self.mean)) > m_threshold:
            self.mean= np.zeros_like(self.mean)
        # If the time step is lower than t_stabilisation,
        # set mean to 0
        if self.t < t_stabilisation:
            self.mean = np.zeros(shape=(self.dimension,1))
        
    
    def evaluate_rewards(self, action_vectors):
        """Evaluate rewards for given action vectors."""
        probabilities = sigmoid(self.mean.T @ action_vectors)
        rewards = action_vectors[1,:] * probabilities
        return rewards.reshape(-1)
    

    def update(self, xk, Y):
        """Calculate current posterior based on all the observations
        including the weighting scheme induced by random walk of the parameters
        """
        if self.xks is None:
            self.xks = xk
            self.Ys = np.array([Y])
        else:
            self.xks = np.hstack((self.xks, xk))
            self.Ys = np.append(self.Ys, Y)
        self.t += 1
        self.stabilise()

        ks = np.linspace(self.t-1, 0, self.t)
        weights = 1/np.sqrt(1+self.f_lambda*ks*self.eta *
                            np.sum(np.abs(self.xks), axis=0))
        wmutxks = weights * (self.mean.T @ self.xks)

        self.cov_0k = self.cov_0 + self.t*self.eta*np.identity(self.dimension)
        self.S_t = np.zeros_like(self.cov_0)
        self.S_t_factors = sigmoid(wmutxks)*(
            1-sigmoid(wmutxks))*(weights**2)
        self.S_t = self.S_t + \
            self.xks @ (self.S_t_factors * self.xks).T
        self.mu_sum = np.zeros_like(self.mean)
        self.mu_sum += np.sum((self.Ys - sigmoid(wmutxks)) * weights
                              * self.xks, axis=1, keepdims=True)
        new_cov = scipy.linalg.inv(scipy.linalg.inv(self.cov_0k) + self.S_t)
        new_mean = new_cov @ (self.S_t @ self.mean + self.mu_sum)
        self.mean = new_mean
        self.cov = new_cov
        return new_mean, new_cov


    # def update(self, xk, Y):
    #     if self.xks is None:
    #         self.xks = xk
    #         self.Ys = np.array([Y])
    #     else:
    #         self.xks = np.hstack((self.xks, xk))
    #         self.Ys = np.append(self.Ys, Y)
    #     self.t += 1
    #     self.stabilise()
    #     mutxks = self.mean.T @ self.xks
    #     self.S_t = np.zeros_like(self.cov_0)
    #     self.S_t_factors = sigmoid(mutxks)*(
    #         1-sigmoid(mutxks))
    #     self.S_t = self.S_t + \
    #         self.xks @ (self.S_t_factors * self.xks).T
    #     self.mu_sum = np.zeros_like(self.mean)
    #     self.mu_sum += np.sum((self.Ys - sigmoid(mutxks))
    #                           * self.xks, axis=1, keepdims=True)
    #     new_cov = scipy.linalg.inv(scipy.linalg.inv(self.cov_0) + self.S_t)
    #     new_mean = new_cov @ (self.S_t @ self.mean + self.mu_sum)
    #     self.mean = new_mean
    #     self.cov = new_cov
    #     return new_mean, new_cov
    
    
    # def iterative_update(self, xk, Y, iterations=5):
    #     """Update model parameters."""
    #     if self.xks is None:
    #         self.xks = xk
    #         self.Ys = np.array([Y])
    #     else:
    #         self.xks = np.hstack((self.xks, xk))
    #         self.Ys = np.append(self.Ys, Y)
    #     self.t += 1
    #     self.stabilise(t_stabilisation=10000)
    #     for iteration in range(iterations):
    #         mutxks = self.mean.T @ self.xks
    #         self.S_t = np.zeros_like(self.cov_0)
    #         self.S_t_factors = sigmoid(mutxks)*(
    #             1-sigmoid(mutxks))
    #         self.S_t = self.S_t + \
    #             self.xks @ (self.S_t_factors * self.xks).T
    #         self.mu_sum = np.zeros_like(self.mean)
    #         self.mu_sum += np.sum((self.Ys - sigmoid(mutxks))
    #                             * self.xks, axis=1, keepdims=True)
    #         new_cov = scipy.linalg.inv(scipy.linalg.inv(self.cov_0) + self.S_t)
    #         new_mean = new_cov @ (self.S_t @ self.mean + self.mu_sum)
    #         self.mean = new_mean
    #         self.cov = new_cov
    #     return new_mean, new_cov


    # def posterior(self, mu):
    #     """Computes the posterior of the model."""
    #     mu = mu.reshape(-1)
    #     posterior = np.sum(self.Ys * (mu @ self.xks) + np.log(1-sigmoid(mu @ self.xks)))
    #     return -posterior

    # def laplace_update(self, xk, Y):
    #     if self.xks is None:
    #         self.xks = xk
    #         self.Ys = np.array([Y])
    #     else:
    #         self.xks = np.hstack((self.xks, xk))
    #         self.Ys = np.append(self.Ys, Y)
    #     self.t += 1
    #     self.stabilise(t_stabilisation=10000)
    #     res = scipy.optimize.minimize(self.posterior, self.mean.reshape(-1), method='nelder-mead',
    #            options={'xatol': 1e-1, 'disp': False})
    #     self.mean = res.x.reshape(-1,1)
    #     mutxks = self.mean.T @ self.xks
    #     self.S_t = np.zeros_like(self.cov_0)
    #     self.S_t_factors = sigmoid(mutxks)*(
    #         1-sigmoid(mutxks))
    #     self.S_t = self.S_t + \
    #         self.xks @ (self.S_t_factors * self.xks).T
    #     new_cov = scipy.linalg.inv(scipy.linalg.inv(self.cov_0) + self.S_t)
    #     self.cov = new_cov
    #     return self.mean, new_cov


    # def get_next_norm(self, xk):
    #     """Computes the max norm of the next covariance matrix based on the chosen action."""
    #     xk = xk.reshape(-1,1)
    #     if self.xks is None:
    #         self.xks_p = xk
    #     else:
    #         self.xks_p = np.hstack((self.xks, xk))
    #     mutxks = self.mean.T @ self.xks_p
    #     self.S_t = np.zeros_like(self.cov_0)
    #     self.S_t_factors = sigmoid(mutxks)*(
    #         1-sigmoid(mutxks))
    #     self.S_t = self.S_t + \
    #         self.xks_p @ (self.S_t_factors * self.xks_p).T
    #     new_cov = scipy.linalg.inv(scipy.linalg.inv(self.cov_0) + self.S_t)
    #     # compute the max norm of the new covatiance matrix
    #     max_norm = np.max(np.abs(new_cov))
    #     return max_norm