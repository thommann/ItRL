import pandas as pd
from matplotlib import pyplot
from Assignment.test import test
from Assignment.train import train
from Assignment.trainQExperienceReplay import train_er


def plot(data, y_range, title, file_name, y_label):
    ema_moves = pd.DataFrame(data).ewm(halflife=1000).mean()

    pyplot.figure()
    pyplot.title(title)
    pyplot.axis([0, len(data), y_range[0], y_range[1]])
    pyplot.xlabel("Nr Episodes")
    pyplot.ylabel(y_label)
    pyplot.plot(ema_moves)
    pyplot.savefig(f"{file_name}.png")
    pyplot.show()


def train_test_plot(strategy="q", beta=0.9999, episodes=100000, hidden=256):
    print(f"Train {strategy}:")
    if strategy == "er":
        nr_moves, total_rewards = train_er(beta=beta, episodes=episodes, hidden=hidden)
    elif strategy == "q" or strategy == "sarsa":
        nr_moves, total_rewards = train(strategy=strategy, beta=beta, episodes=episodes, hidden=hidden)
    else:
        print(f"INVALID STRATEGY: {strategy}")
        return

    plot(total_rewards, [-1, 1.1], f"{strategy}: Rewards per Game while training", f"{strategy}-{hidden}-train-rewards",
         "Rewards")
    plot(nr_moves, [0, 20], f"{strategy}: Number of moves per Game while training", f"{strategy}-{hidden}-train-moves",
         "Nr moves")
    print()

    print(f"Test {strategy}:")
    nr_moves_test, total_rewards_test = test(strategy=strategy)
    plot(total_rewards_test, [0.8, 1.1], f"{strategy}: Rewards per Game", f"{strategy}-{hidden}-test-rewards",
         "Rewards")
    plot(nr_moves_test, [0, 20], f"{strategy}: Number of moves per Game", f"{strategy}-{hidden}-test-moves",
         "Nr moves")
    print()
