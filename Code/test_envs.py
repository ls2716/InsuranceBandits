import unittest

import numpy as np
from envs import Calvano, InsuranceMarket, InsuranceMarketCt


class TestCalvano(unittest.TestCase):

    def setUp(self) -> None:
        self.env = Calvano(np.array([1., 1.]), 0.1, 1., np.array([0., 0.]))

    def test_equal_rewards(self):
        rewards = self.env.step(np.array([1., 1.]))
        self.assertEqual(rewards[0, 0], rewards[0, 1])

    def test_rewards(self):
        rewards = self.env.step(np.array([0.7, 0.5]))
        self.assertAlmostEqual(0.44039854, rewards[0, 1])


class TestInsurance(unittest.TestCase):

    def setUp(self) -> None:
        self.env = InsuranceMarket(2, 0.2, 0., 1., 0.2)
        np.random.seed(0)

    def test_rewards(self):
        rewards = self.env.step(
            np.array([[1., 1.], [1., 1.], [1., 1.], [1., 1.]]))
        self.assertEqual(rewards.shape[0], 4)
        self.assertEqual(rewards.shape[1], 2)
        self.assertAlmostEqual(rewards[2, 1], 0.80454442)


class TestInsuranceCt(unittest.TestCase):

    def setUp(self) -> None:
        self.env = InsuranceMarketCt(2, 0.2, 0., 1., 0.2)
        np.random.seed(0)

    def test_rewards(self):
        rewards = self.env.step(
            np.array([[1., 1.], [1., 1.], [0.9, 1.], [1., 0.8]]))
        self.assertEqual(rewards.shape[0], 4)
        self.assertEqual(rewards.shape[1], 2)
        self.assertAlmostEqual(rewards[3, 1], 0.8)


if __name__ == '__main__':
    unittest.main()
