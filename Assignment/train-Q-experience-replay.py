from Assignment.Chess import Chess
from Assignment.Experience import Experience
from Assignment.Network import Network, epsilon_greedy_policy, pickle_network

from matplotlib import pyplot
import numpy as np


def train():
    network = Network(512, 58, 32)
    episodes = 20000
    epsi = 0.4
    gamma = 0.7
    nr_steps = []
    Qvalues = 2
    count_2 = 0
    _rewards=[]
    experiences = []
    batch_size = 200
    relevant_histories = 600
    for episode in range(episodes):
        if episode in [10000, 13000, 15000, 17000]:
            network.eta /= 3
        if episode % 100 == 0:
            print(f"\rEpi: {episode}, epsi: {epsi:.3f}, avg. stepsi: {count_2 / 100:.2f}, learni: {network.eta:.4f}",
                  end="")
            epsi *= 0.96
            count_2 = 0

        chess = Chess()
        count_1 = 0
        while True:
            count_1 += 1
            count_2 += 1
            chess_prime = chess.clone()
            Qvalues, H = network.forward(chess.state)
            Qvalues -= (1 - chess.get_valid_actions()) * 100000
            action = epsilon_greedy_policy(np.array([Qvalues]), epsi).T
            reward = chess.do_action(action)
            _rewards.append(reward)
            expi = Experience(chess_prime.state, chess.state, action, reward)
            experiences.append(expi)
            if len(experiences) > relevant_histories:
                experiences = experiences[1:]

                training_set = np.random.choice(experiences,batch_size, False)

                state_before = np.hstack([experience.state_before for experience in training_set])
                Qvalues_before, H_before = network.forward(state_before)
                actions = np.hstack([experience.action for experience in training_set])
                Qvalues_played = Qvalues_before * actions
                Qvalues_after, H_after = network.forward(np.hstack([experience.state_after for experience in training_set]))

                Q_values_after_played = np.max(Qvalues_after, 0)

                rewards = np.hstack([experience.reward for experience in training_set])

                target = (rewards + gamma * Q_values_after_played) * actions
                network.descent(state_before, target, H_before, Qvalues_played)
            if chess.done:
                nr_steps.append(count_1)
                break
            chess.move_b()

    pyplot.figure()
    pyplot.title("Train: # Moves")
    pyplot.plot(moving_average(nr_steps, 500))
    pyplot.show()

    pyplot.figure()
    pyplot.title("Train: Reward")
    pyplot.plot(moving_average(_rewards, 500))
    pyplot.show()

    pickle_network(network)


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


if __name__ == "__main__":
    train()
