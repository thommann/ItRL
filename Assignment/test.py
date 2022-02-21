import numpy as np

from Assignment.Chess import Chess
from Assignment.Network import depickle, epsilon_greedy_policy


def test(
        strategy="q",
        hidden=256,
        episodes=10000
):
    network = depickle(f"{strategy}-{hidden}.pcl")
    nr_moves = []
    total_rewards = []
    stuck_runs = 0
    for i in range(episodes):
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
                    stuck_runs += 1
                    if stuck_runs % 20 == 0:
                        print(f"\rStuck: {stuck_runs}!")
                nr_moves.append(count)
                total_rewards.append(total_reward)
                break
            test_chess.move_b()

    print()
    print(f"Stuck runs: {stuck_runs}")
    return nr_moves, total_rewards


if __name__ == "__main__":
    test()
