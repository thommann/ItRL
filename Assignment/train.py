from Assignment.Chess import Chess
from Assignment.Network import Network, epsilon_greedy_policy, pickle_network

from matplotlib import pyplot
import numpy as np


def train():
<<<<<<< HEAD
    network = Network(1024, 58, 32)
    episodes = 300000
=======
    network = Network(256, 58, 32)
    episodes = 10000
>>>>>>> ac46be9ac11471135a47ffa6b6964bc5988d8f91
    epsi = 0.4
    gamma = 0.7
    nr_moves = []
    rewards = []
    count_100_episodes = 0
    for episode in range(episodes):
<<<<<<< HEAD
        if episode == 150000 or episode == 200000:
            network.eta /= 2
=======
>>>>>>> ac46be9ac11471135a47ffa6b6964bc5988d8f91
        if episode % 100 == 0:
            print(f"\repisode: {episode}, "
                  f"epsilon: {epsi:.3f}, "
                  f"moves: {count_100_episodes / 100:.2f}, "
                  f"eta: {network.eta:.4f}",
                  end="")
            count_100_episodes = 0

        chess = Chess()
        Qvalues, H = network.forward(chess.state)
        Qvalues -= (1 - chess.get_valid_actions()) * 100000
        action = epsilon_greedy_policy(np.array([Qvalues]), epsi).T
        count_episode = 0
        total_reward = 0
        while True:
            count_episode += 1
            count_100_episodes += 1
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
<<<<<<< HEAD
            H = H_prime
=======
            total_reward += reward

>>>>>>> ac46be9ac11471135a47ffa6b6964bc5988d8f91
            if chess.done:
                nr_moves.append(count_episode)
                rewards.append(total_reward/count_episode)
                break
            chess.move_b()

    pyplot.figure()
    pyplot.title("Train: # Moves")
    pyplot.plot(moving_average(nr_moves, 500))
    pyplot.show()

    pyplot.figure()
    pyplot.title("Train: Reward")
    pyplot.plot(moving_average(rewards, 500))
    pyplot.show()

    pickle_network(network)


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


if __name__ == "__main__":
    train()
