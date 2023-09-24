# Implementation of the bandit environment models


# Import libraries
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import scipy

from utils import get_logger

logger = get_logger(__name__)
# logger.setLevel('DEBUG')


class BaseModel(object):
    """Base model object"""

    def __init__(self, candidate_margins) -> None:
        self.actions = candidate_margins.reshape(-1)
        self.n_actions = self.actions.shape[0]
        pass

    def reset(self) -> None:
        pass

    def update(self, action_index, observation) -> None:
        pass

    def get_expected_rewards(self):
        pass

    def set_action_vectors(self):
        pass

    def get_quantiles(self, quantile):
        pass

    def sample_reward_TS(self):
        pass


def sigmoid(x):
    return torch.special.expit(x)


class LogisticModelLaplace(BaseModel):
    """Defines the logistic regression model for bandits
    with iterative update for the mean based on of all the observations."""

    def __init__(self, candidate_margins, dimension=2, variance=10**3) -> None:
        """Initialisation function."""
        super().__init__(candidate_margins=torch.Tensor(candidate_margins))
        self.dimension = dimension
        self.variance = variance
        self.reset()
        self.set_action_vectors()

    def reset(self):
        """Reset model."""
        self.mean = torch.zeros(size=(self.dimension, 1))
        self.cov = torch.eye(self.dimension)*self.variance
        self.cov_0 = torch.eye(self.dimension)*self.variance
        self.xks = None
        self.Ys = None
        self.t = 0

    def stabilise(self, m_threshold=100, cov_threshold=50):
        """A stabilistation function for the model."""

        # If the mean is too large, set it to 0
        if torch.max(torch.abs(self.mean)) > m_threshold:
            self.mean = torch.zeros_like(self.mean)
        if torch.max(torch.abs(self.cov)) > cov_threshold:
            self.mean = torch.zeros_like(self.mean)

    def sample_rewards(self, action_vectors, no_samples):
        """Sample rewards for given action vectors."""
        # Sample parameters from current posterior
        mean_samples = torch.random.multivariate_normal(
            self.mean.reshape(-1), self.cov, size=no_samples)
        probabilities = sigmoid(mean_samples @ action_vectors)
        rewards = action_vectors[1, :] * probabilities
        return rewards

    def update(self, action_index, reward, observation):
        xk = self.action_vectors[:, action_index].reshape(-1, 1)
        Y = observation
        if self.xks is None:
            self.xks = xk
            self.Ys = torch.tensor([Y], dtype=torch.float32)
        else:
            self.xks = torch.hstack((self.xks, xk))
            self.Ys = torch.cat((self.Ys, torch.tensor([Y])))
        self.t += 1
        self.stabilise()

        # Find the posterior mean
        new_mean = self.get_mean_scipy()
        new_mean = torch.tensor(new_mean, dtype=torch.float32)

        # Compute the hessian of log prob using the new mean
        new_inv_cov = - \
            torch.func.hessian(self.log_prob)(new_mean.reshape(-1))
        # print(f'Hessian {new_inv_cov}')
        new_cov = torch.linalg.inv(new_inv_cov)
        self.mean = new_mean
        self.cov = new_cov
        return new_mean, new_cov

    def log_prob(self, theta):
        """Log probability of the parameter vector theta."""
        # Add log posterior of the normal prior
        log_prob = -0.5 * theta.T @ torch.linalg.inv(self.cov_0) @ theta
        # Compute sigmoids
        probabilities = sigmoid(theta.T @ self.xks)
        # Compute log likelihoods
        likelihoods = self.Ys * torch.log(probabilities) + \
            (1-self.Ys) * torch.log(1-probabilities)
        # Add logs of likelihoods
        log_prob += torch.sum(likelihoods)
        return log_prob

    def log_prob_scipy(self, theta):
        """Log probability of the parameter vector theta."""
        theta = theta.reshape(-1, 1)
        xks = self.xks.detach().numpy()
        Ys = self.Ys.detach().numpy()
        # Add log posterior of the normal prior
        log_prob = -0.5 * theta.T @ scipy.linalg.inv(self.cov_0) @ theta
        # Compute sigmoids
        probabilities = scipy.special.expit(theta.T @ xks)
        # Compute log likelihoods
        likelihoods = Ys * np.log(probabilities) + \
            (1-Ys) * np.log(1-probabilities)
        # Add logs of likelihoods
        log_prob += np.sum(likelihoods)
        return log_prob

    def get_mean_scipy(self):
        """Compute the mean using scipy minimize"""
        theta_0 = self.mean.detach().numpy().reshape(-1)
        res = scipy.optimize.minimize(
            lambda theta: -self.log_prob_scipy(theta), theta_0, tol=1e-2, options={'disp': False})
        return res.x.reshape(-1, 1)

    def get_mean(self):
        theta_0 = torch.tensor(torch.zeros_like(self.mean), requires_grad=True)

        def loss_function(theta):
            """Loss functio for the optimization"""
            return -self.log_prob(theta)

        # logger.info('Optimizing the loss function')
        # Optimize the loss function
        optimizer = torch.optim.Adam([theta_0], lr=1.)
        for i in range(100):
            optimizer.zero_grad()
            loss = loss_function(theta_0)
            loss.backward()
            optimizer.step()
        # logger.info(f'Optimal theta found {theta_0}')
        return theta_0.reshape(-1, 1)

    def get_expected_rewards(self):
        probabilities = sigmoid(self.mean.T @ self.action_vectors)
        rewards = self.action_vectors[1, :] * probabilities
        return rewards.reshape(-1).detach().numpy()

    def set_action_vectors(self):
        self.action_vectors = torch.ones((self.dimension, self.n_actions))
        for i in range(1, self.dimension):
            self.action_vectors[i, :] = self.actions * \
                self.action_vectors[i-1, :]
        logger.info(f'Action vectors: \n {self.action_vectors}')

    def get_quantiles(self, quantile, size=1000):
        # Generate parameter samples
        mean_samples = np.random.multivariate_normal(
            mean=self.mean.reshape(-1), cov=self.cov, size=size)
        # Compute expected reward samples
        probabilities = sigmoid(mean_samples @ self.action_vectors)
        reward_samples = self.action_vectors[1, :] * probabilities
        # Return quantile of the reward samples
        return torch.quantile(reward_samples, quantile, axis=0)

    def sample_reward_TS(self):
        # Generate parameter samples
        mean_sample = torch.random.multivariate_normal(
            mean=self.mean.reshape(-1), cov=self.cov, size=1)
        # Compute expected reward samples
        probabilities = sigmoid(mean_sample @ self.action_vectors)
        reward_sample = self.action_vectors[1, :] * probabilities
        # Return quantile of the reward samples
        return reward_sample.reshape(-1)


# class NonStationaryLogisticModel(StationaryLogisticModel):
#     """Implementation of non-stationary logistic model.

#     Uses either discounting or sliding window approach."""

#     # Initialisation function
#     def __init__(self, candidate_margins, dimension=2, variance=10**3, method='discounting',
#                  gamma=0.9, tau=100) -> None:
#         super().__init__(candidate_margins=candidate_margins,
#                          dimension=dimension, variance=variance)
#         self.method = method
#         self.method = method
#         if method == 'discounting':
#             self.update = self.update_discounting
#             self.gamma = gamma
#         elif method == "sliding_window":
#             self.update = self.update_sliding_window
#             self.tau = tau
#         else:
#             raise ValueError(
#                 'Invalid method for nonstationarity. Choose "discounting" or "sliding_window".')
#         self.reset()

#     def update_discounting(self, action_index, reward, observation):
#         xk = self.action_vectors[:, action_index].reshape(-1, 1)
#         Y = observation
#         if self.xks is None:
#             self.xks = xk
#             self.Ys = np.array([Y])
#             self.weights = np.array([1])
#         else:
#             self.xks = np.hstack((self.xks, xk))
#             self.Ys = np.append(self.Ys, Y)
#             self.weights = np.append(self.weights*self.gamma, 1)
#         self.t += 1
#         self.stabilise()
#         mutxks = self.mean.T @ self.xks
#         self.S_t = np.zeros_like(self.cov_0)
#         self.S_t_factors = sigmoid(mutxks)*(
#             1-sigmoid(mutxks))*self.weights
#         self.S_t = self.S_t + \
#             self.xks @ (self.S_t_factors * self.xks).T
#         self.mu_sum = np.zeros_like(self.mean)
#         self.mu_sum += np.sum(self.weights * (self.Ys - sigmoid(mutxks))
#                               * self.xks, axis=1, keepdims=True)
#         new_cov = scipy.linalg.inv(scipy.linalg.inv(
#             self.cov_0) + self.S_t)
#         new_mean = new_cov @ (self.S_t @ self.mean + self.mu_sum)
#         self.mean = new_mean
#         self.cov = new_cov
#         return new_mean, new_cov

#     def update_sliding_window(self, action_index, reward, observation):
#         xk = self.action_vectors[:, action_index].reshape(-1, 1)
#         Y = observation
#         if self.xks is None:
#             self.xks = xk
#             self.Ys = np.array([Y])
#         else:
#             self.xks = np.hstack((self.xks, xk))[:, -self.tau:]
#             self.Ys = np.append(self.Ys, Y)[-self.tau:]
#         self.t += 1
#         self.stabilise()
#         mutxks = self.mean.T @ self.xks
#         self.S_t = np.zeros_like(self.cov_0)
#         self.S_t_factors = sigmoid(mutxks)*(
#             1-sigmoid(mutxks))
#         self.S_t = self.S_t + \
#             self.xks @ (self.S_t_factors * self.xks).T
#         self.mu_sum = np.zeros_like(self.mean)
#         self.mu_sum += np.sum((self.Ys - sigmoid(mutxks))
#                               * self.xks, axis=1, keepdims=True)
#         new_cov = scipy.linalg.inv(scipy.linalg.inv(self.cov_0) + self.S_t)
#         new_mean = new_cov @ (self.S_t @ self.mean + self.mu_sum)
#         self.mean = new_mean
#         self.cov = new_cov
#         return new_mean, new_cov


# def plot_logistic(mean, x_true, y_true, title, foldername, filename, show_plot=True):
#     """Plot logistic function for given mean."""
#     x_max = np.max(x_true)
#     x_min = np.min(x_true)
#     x = np.linspace(x_min-0.2, x_max+0.2, 100)
#     y = sigmoid(mean[0] + mean[1]*x)
#     plt.plot(x, y, label='Fitted model')
#     plt.scatter(x_true, y_true, label='True values')
#     plt.xlabel('Margin')
#     plt.ylabel('Probability')
#     plt.legend()
#     plt.title(title)
#     plt.savefig(os.path.join(foldername, filename))
#     if show_plot:
#         plt.show()
#     plt.close()


# def plot_classic(mean, x_true, y_true, title, foldername, filename, show_plot=True):
#     """Plot the learned rewards"""
#     plt.scatter(x_true, mean, label='Fitted model')
#     plt.scatter(x_true, y_true, label='True values')
#     plt.xlabel('Margin')
#     plt.ylabel('Reward')
#     plt.legend()
#     plt.title(title)
#     plt.savefig(os.path.join(foldername, filename))
#     if show_plot:
#         plt.show()
#     plt.close()


# if __name__=="__main__":
