import numpy as np
from matplotlib import pyplot

from Assignment.Chess import Chess
from Assignment.Network import depickle, epsilon_greedy_policy
from Assignment.train import moving_average


def test():
    network = depickle()
    nr_moves = []
    rewards = []
    for i in range(10000):
        test_chess = Chess()
        count = 0
        if i % 100 == 0:
            print(f"\r{i}", end="")
        total_reward = 0
        while True:
            count += 1
            test_Qvalues, _ = network.forward(test_chess.state)
            test_Qvalues -= (1 - test_chess.get_valid_actions()) * 100000
            a = epsilon_greedy_policy(np.array([test_Qvalues]), 0).T
            reward = test_chess.do_action(a)
            total_reward += reward
            if test_chess.done or count > 1000:
                nr_moves.append(count)
                rewards.append(total_reward / count)
                break
            test_chess.move_b()

    pyplot.figure()
    pyplot.title("Test: # Moves")
    pyplot.plot(moving_average(nr_moves, 500))
    pyplot.show()

    pyplot.figure()
    pyplot.title("Test: Reward")
    pyplot.plot(moving_average(rewards, 500))
    pyplot.show()


if __name__ == "__main__":
    test()
