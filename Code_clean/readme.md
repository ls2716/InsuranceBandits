# Bandits for Insurance Pricing

In this folder, we have put a clean code for insurance pricing project with bandits.
The aim of the code of this project is to investigate:
- how to add an environment model to insurance pricing bandits and how does it influence the interactions between bandits 
- how to add non-stationarity to the model and how it influences the interactions

## Code structure

The bandit algorithms should be based on a base bandit class which then extend to specific method of action selection. Furthermore, each bandit should allow for basing itself on a possibly non-stationary environment model.

### Environment model

The environment model used will be logistic model with two parameters which should well approximate the probability curves of the real underlying environment composed through analytical calculation of offer acceptance probability.

