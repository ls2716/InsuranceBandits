## Bandits in multi-agent systems

This repository contains code for the preject regarding learning of one-arm bandits in
multi-agent systems.

In this case, the environments consist of competetive environments with no state. That is,
at each step, each agent submits their action and each agent receives a reward which is a
function of all agents' actions.

envs.py file contains objects representing the environments.

### Environments

The environments are as follows:

1. Market environment where at each step, each agent receives their estimate of the true cost
and based on that estimate submit a price. The agent with the lowest price wins the trade, if the price is lower than the reservation price of the customer, the agent receives a reward equal to the price minus the true cost.

2. Same as above but reward is price minus the estimated cost (Nash's note).

3. Market model from Calvano's paper, where the demand is a function of all player's qualities and prices.

### Agents

The agents are to be defined but all consist of single player bandit algorithms with modifications for exploration.

