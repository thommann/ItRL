from Assignment.Chess import Chess
from Assignment.Network import Network, epsilon_greedy_policy, pickle_network

import numpy as np


def train(
        strategy="sarsa",
        hidden=256,
        episodes=100000,
        epsilon=0.4,
        gamma=0.7,
        beta=1,
        eta_decay=1,
):
    nr_moves = []
    total_rewards = []
    network = Network(hidden, 58, 32)
    for episode in range(episodes):
        if episode > 0 and episode % 100 == 0:
            print(f"\repisode: {episode}, "
                  f"epsilon: {epsilon:.3f}, "
                  f"moves: {np.average(nr_moves[episode-100:]):.2f}, "
                  f"eta: {network.eta:.4f}, "
                  f"W1: {np.min(network.W1):.4f}, {np.max(network.W1):.4f}, "
                  f"W2: {np.min(network.W2):.4f}, {np.max(network.W2):.4f}",
                  end="")

        network.eta *= eta_decay
        epsilon *= beta
        chess = Chess()
        Qvalues, H = network.forward(chess.state)
        Qvalues -= (1 - chess.get_valid_actions()) * 100000
        action = epsilon_greedy_policy(np.array(Qvalues), epsilon).T

        count_episode = 0
        total_reward = 0
        while True:
            count_episode += 1
            chess_prime = chess.clone()
            reward = chess.do_action(action)
            total_reward += reward

            if chess.done:
                output = Qvalues * action
                target = reward * action
                network.descent(chess_prime.state, target, H, output)

                nr_moves.append(count_episode)
                total_rewards.append(total_reward)
                break

            chess.move_b()
            Qvalues_prime, H_prime = network.forward(chess.state)
            Qvalues_prime -= (1 - chess.get_valid_actions()) * 200000

            if strategy == "sarsa":
                next_action = epsilon_greedy_policy(np.array(Qvalues_prime), epsilon).T
            elif strategy == "q":
                next_action = epsilon_greedy_policy(np.array(Qvalues_prime), 0).T
            else:
                print(f"Illegal strategy: {strategy}!")
                break

            Q_prime = Qvalues_prime[np.argmax(next_action)]
            target = (reward + gamma * Q_prime) * action
            output = Qvalues * action
            network.descent(chess_prime.state, target, H, output)

            Qvalues = Qvalues_prime
            H = H_prime
            if strategy == "sarsa":
                action = next_action
            elif strategy == "q":
                action = epsilon_greedy_policy(np.array(Qvalues_prime), epsilon).T
            else:
                print(f"Illegal strategy: {strategy}!")
                break

    pickle_network(network, f"{strategy}-{hidden}.pcl")
    return nr_moves, total_rewards


if __name__ == "__main__":
    train()
