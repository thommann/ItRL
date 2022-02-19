from Assignment.Chess import Chess
from Assignment.Network import Network, epsilon_greedy_policy, pickle_network

from matplotlib import pyplot
import numpy as np


def train():
    network = Network(1024, 58, 32)
    episodes = 300000
    epsi = 0.4
    gamma = 0.7
    nr_steps = []
    Qvalues = 2
    count_2 = 0
    for episode in range(episodes):
        if episode == 150000 or episode == 200000:
            network.eta /= 2
        if episode % 100 == 0:
            print(f"\rEpi: {episode}, epsi: {epsi:.3f}, avg. stepsi: {count_2 / 100:.2f}, learni: {network.eta:.4f}",
                  end="")
            epsi *= 0.997
            count_2 = 0

        chess = Chess()
        Qvalues, H = network.forward(chess.state)
        Qvalues -= (1 - chess.get_valid_actions()) * 100000
        action = epsilon_greedy_policy(np.array([Qvalues]), epsi).T
        count_1 = 0
        while True:
            count_1 += 1
            count_2 += 1
            chess_prime = chess.clone()
            reward = chess.do_action(action)
            Qvalues_prime, H_prime = network.forward(chess.state)
            Qvalues_prime -= (1 - chess.get_valid_actions()) * 100000
            next_action = epsilon_greedy_policy(np.array([Qvalues_prime]), epsi).T
            Q_prime = Qvalues_prime[np.argmax(next_action)]
            target = (reward + gamma * Q_prime) * action
            output = Qvalues * action
            network.descent(chess_prime.state, target, H, output)
            action = next_action
            Qvalues = Qvalues_prime
            H = H_prime
            if chess.done:
                nr_steps.append(count_1)
                break
            chess.move_b()
    pyplot.plot(moving_average(nr_steps, 500))
    pyplot.show()

    pickle_network(network)


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


if __name__ == "__main__":
    train()
