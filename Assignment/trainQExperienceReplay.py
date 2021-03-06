from Assignment.Chess import Chess
from Assignment.Experience import Experience
from Assignment.Network import Network, epsilon_greedy_policy, pickle_network
import numpy as np


def train_er(
        hidden=256,
        episodes=2000,
        epsilon=0.4,
        gamma=0.7,
        eta_decay=1,
        beta=1,
):
    network = Network(hidden, 58, 32)
    _rewards = []
    nr_moves = []
    experiences = []
    batch_size = 50
    relevant_histories = 300
    for episode in range(episodes):
        if episode > 0 and episode % 100 == 0:
            print(f"\rEpi: {episode},"
                  f" epsi: {epsilon:.3f}, "
                  f"avg. stepsi: {np.average(nr_moves[episode - 100:]):.2f}, "
                  f"learni: {network.eta:.4f} "
                  f"W1: {np.min(network.W1):.4f}, {np.max(network.W1):.4f}, "
                  f"W2: {np.min(network.W2):.4f}, {np.max(network.W2):.4f}",
                  end="")

        epsilon *= beta
        network.eta *= eta_decay
        chess = Chess()
        count_1 = 0
        total_reward = 0
        while True:
            count_1 += 1
            chess_prime = chess.clone()

            Qvalues, H = network.forward(chess.state)
            Qvalues -= (1 - chess.get_valid_actions()) * 100000
            action = epsilon_greedy_policy(np.array(Qvalues), epsilon).T

            reward = chess.do_action(action)
            total_reward += reward
            expi = Experience(chess_prime.state, chess.state, action, reward)
            experiences.append(expi)
            if len(experiences) > relevant_histories:
                experiences = experiences[1:]

                training_set = np.random.choice(experiences, batch_size, False)

                state_before = np.hstack([experience.state_before for experience in training_set])
                Qvalues_before, H_before = network.forward(state_before)
                actions = np.hstack([experience.action for experience in training_set])
                Qvalues_played = Qvalues_before * actions
                Qvalues_after, H_after = network.forward(
                    np.hstack([experience.state_after for experience in training_set]))

                Q_values_after_played = np.max(Qvalues_after, 0)

                rewards = np.hstack([experience.reward for experience in training_set])

                target = (rewards + gamma * Q_values_after_played) * actions
                network.descent(state_before, target, H_before, Qvalues_played)
            if chess.done:
                nr_moves.append(count_1)
                _rewards.append(total_reward)
                break
            chess.move_b()

    pickle_network(network, f"er-{hidden}.pcl")
    return nr_moves, _rewards


if __name__ == "__main__":
    train_er()
