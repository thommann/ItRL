import numpy as np
from matplotlib import pyplot
import pandas as pd

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
        while True:
            count += 1
            test_Qvalues, _ = network.forward(test_chess.state)
            test_Qvalues -= (1 - test_chess.get_valid_actions()) * 100000
            a = epsilon_greedy_policy(np.array([test_Qvalues]), 0).T
            reward = test_chess.do_action(a)
            if test_chess.done or count > 100:
                if count > 100:
                    print("Bogo hit")
                nr_moves.append(count)
                rewards.append(reward)
                break
            test_chess.move_b()

    ema_moves = pd.DataFrame(nr_moves).ewm(halflife=1000).mean()
    ema_rewards = pd.DataFrame(rewards).ewm(halflife=1000).mean()

    pyplot.figure()
    pyplot.title("Test: # Moves")
    pyplot.plot(ema_moves)
    pyplot.show()
    pyplot.savefig("test_moves_sarsa.png")

    pyplot.figure()
    pyplot.title("Test: Reward")
    pyplot.plot(ema_rewards)
    pyplot.show()
    pyplot.savefig("test_reward_sarsa.png")


if __name__ == "__main__":
    test()
