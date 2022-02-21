import numpy as np

from Assignment.Chess import Chess
from Assignment.Network import depickle, epsilon_greedy_policy


def test(filename="q-experience-replay-256.pcl"):
    network = depickle(filename)
    nr_moves = []
    total_rewards = []
    bogos = 0
    for i in range(10000):
        test_chess = Chess()
        total_reward = 0
        count = 0
        if i % 100 == 0:
            print(f"\r{i}", end="")
        while True:
            count += 1
            test_Qvalues, _ = network.forward(test_chess.state)
            test_Qvalues -= (1 - test_chess.get_valid_actions()) * 100000
            a = epsilon_greedy_policy(np.array(test_Qvalues), 0).T
            reward = test_chess.do_action(a)
            total_reward += reward
            if test_chess.done or count > 100:
                if count > 100:
                    bogos += 1
                    if bogos % 20 == 0:
                        print(f"Bogo hit: {bogos}")
                nr_moves.append(count)
                total_rewards.append(total_reward)
                break
            test_chess.move_b()

    return nr_moves, total_rewards


if __name__ == "__main__":
    test()
