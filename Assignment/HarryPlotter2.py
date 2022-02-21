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


# SARSA
# nr_moves_sarsa, rewards_sarsa = train("sarsa")
# plot(rewards_sarsa, [-1,1.1], "Sarsa: Rewards per Game while training", "sarsa-256-hidden-layers-train-rewards", "Rewards")
# plot(nr_moves_sarsa, [0,20], "Sarsa: Number of moves per Game while training", "sarsa-256-hidden-layers-train-moves", "Nr moves")
#
# nr_moves_sarsa_test, rewards_sarsa_test = test("sarsa-256.pcl")
# plot(rewards_sarsa_test, [0.8,1.1], "Sarsa: Rewards per Game", "sarsa-256-hidden-layers-test-rewards", "Rewards")
# plot(nr_moves_sarsa_test, [0,20], "Sarsa: Number of moves per Game", "sarsa-256-hidden-layers-test-moves", "Nr moves")

# Q-learning
nr_moves_q, rewards_q = train("q")
plot(rewards_q, [-1, 1.1], "Q: Rewards per Game while training", "q-256-hidden-layers-train-rewards", "Rewards")
plot(nr_moves_q, [0, 20], "Q: Number of moves per Game while training", "q-256-hidden-layers-train-moves", "Nr moves")

nr_moves_q_test, rewards_per_move_q_rest = test("q-256.pcl")
plot(rewards_per_move_q_rest, [0.8, 1.1], "Q: Rewards per Game", "q-256-hidden-layers-test-rewards", "Rewards")
plot(nr_moves_q_test, [0, 20], "Q: Number of moves per Game", "q-256-hidden-layers-test-moves", "Nr moves")
#
#
# # Q-learning with experience replay
# nr_moves_q_er, rewards_q_er = train_er()
# plot(rewards_q_er, [-1,1.1], "Q + Experience Replay: Rewards per Game while training", "q-er-256-hidden-layers-train-rewards", "Rewards")
# plot(nr_moves_q_er, [0,20], "Q + Experience Replay: Number of moves per Game while training", "q-er-256-hidden-layers-train-moves", "Nr moves")
#
# nr_moves_q_er_test, rewards_per_move_q_er_rest = test("q-experience-replay-256.pcl")
# plot(rewards_per_move_q_er_rest, [-1,1.1], "Q + Experience Replay: Rewards per Game", "q-er-256-hidden-layers-test-rewards", "Rewards")
# plot(nr_moves_q_er_test, [0,20], "Q + Experience Replay: Number of moves per Game", "q-er-256-hidden-layers-test-moves", "Nr moves")
