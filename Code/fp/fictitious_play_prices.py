from model import reward
import numpy as np
import matplotlib.pyplot as plt


# Set random seed
seed = 0
np.random.seed(seed)


# Initialize game parameters
sigma = 0.3
rho = 0
tau = 0.2
# Initialize customer margin
S_c = 5
# Initialize number of players
N = 2
# Initialize margins
S_0 = np.random.random(size=(N, N))*1
new_S = np.zeros_like(S_0)

# Initialize optimisation range
S_i = np.linspace(0, 1., 501)

# Initialize number of steps
T = 200
# Initialize history array
history_S = np.zeros(shape=(N, N, T+1))

# Initialize scale of errors
error_scales = np.ones(shape=(N, N))*np.sqrt(2*sigma**2*(1-rho))
for i in range(N):
    error_scales[i, i] = 0


def step(i, S, S_c, sigma, rho, tau):
    for player in range(N):
        Sm = []
        for other_player in range(N):
            if player != other_player:
                # print(player, other_player)
                Sm.append(S[other_player, player])
        player_reward = reward(S_i, Sm, S_c, sigma, rho, tau)
        response = S_i[np.argmax(player_reward)]
        for other_player in range(N):
            new_S[player, other_player] = response
    errors = np.random.normal(0, error_scales)
    S[:, :] = S[:, :] + 1/(i+1) * (
        new_S[:, :] +
        errors - S[:, :]
    )
    # print(new_S)
    # print(errors)
    # print()


for i in range(0, T):
    print(S_0)
    history_S[:, :, i] = S_0[:, :]
    step(i, S_0, S_c, sigma, rho, tau)
history_S[:, :, T] = S_0[:, :]


nash_dict = {
    2: 0.532,
    3: 0.452
}
nash = nash_dict[N]


fig = plt.figure(figsize=(8, 6))
for player in range(N):
    plt.semilogy(
        np.abs(history_S[player, player, :]-nash) +
        10**(-10), label=f"player {player}")
plt.legend()
plt.xlabel('step')
plt.ylabel('deviation from Nash')
plt.title(f"Fictitious play with prices for {N} players seed {seed}")
plt.grid()
plt.savefig(f'fp_prices_{N}.png')
plt.close()
