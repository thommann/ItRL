import numpy as np
from matplotlib import pyplot

from Assignment.Chess import Chess
from Assignment.Network import depickle, epsilon_greedy_policy
from Assignment.train import moving_average


def test():
    network = depickle()
    nr_steps = []
    rewards = []
    for i in range(10000):
        test_chess = Chess()
        count = 0
        if i % 100 == 0:
            print(f"\r{i}", end="")
        while True:
            count += 1
            test_Qvalues, _ = network.forward(test_chess.state)
            test_Qvalues -= (1 - test_chess.get_valid_actions()) * 100000
            a = epsilon_greedy_policy(np.array([test_Qvalues]), 0).T
            reward = test_chess.do_action(a)
            rewards.append(reward)
            if test_chess.done:
                nr_steps.append(count)
                break
            if count > 20:
                print("Bogo hit")
                nr_steps.append(count)
                break
            test_chess.move_b()

    pyplot.figure()
    pyplot.title("Test: # Moves")
    pyplot.plot(moving_average(nr_steps, 2000))
    pyplot.show()

    pyplot.figure()
    pyplot.title("Test: Reward")
    pyplot.plot(moving_average(rewards, 2000))
    pyplot.show()


if __name__ == "__main__":
    test()
