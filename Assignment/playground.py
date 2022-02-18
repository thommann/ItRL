import random

import numpy

from Assignment.Network import Network, epsilon_greedy_policy

from matplotlib import pyplot
import numpy as np
#
# X, T = data.load_categorical_digits()
#
# # Parameters
# B = 100
# K = 10
# eta = 0.01
# epochs = 10000
# mu = 0.5
#
network = Network(30, 58, 32)
#
# loss = network.learn(X, T, K, eta=eta, epochs=epochs, B=B, mu=mu)
#
# pyplot.figure()
#
# ax1 = pyplot.gca()
# ax2 = ax1.twinx()
# ax1.set_xlabel("Epoch")
#
# ax1.loglog(loss[:, 0], color="tab:blue")
# ax1.tick_params(axis="y", labelcolor="tab:blue")
# ax1.set_ylabel("Loss", color="tab:blue")
#
# ax2.semilogx(loss[:, 1], color="tab:red")
# ax2.tick_params(axis="y", labelcolor="tab:red")
# ax2.set_ylabel("Accuracy", color="tab:red")
#
# pyplot.show()

class Chess:
    def __init__(self, initialize = True):
        self.w_king = None
        self.w_queen = None
        self.b_king = None
        if initialize:
            self.initialize_state()
        self.done = False


    def initialize_state(self):
        self.w_king = np.zeros((4, 4))
        self.w_queen = np.zeros((4, 4))
        self.b_king = np.zeros((4, 4))

        w_king_idx = (np.random.randint(0, 4), np.random.randint(0, 4))
        self.w_king[w_king_idx] = 1

        while True:
            w_queen_idx = (np.random.randint(0, 4), np.random.randint(0, 4))
            if w_queen_idx != w_king_idx:
                break
        self.w_queen[w_queen_idx] = 1

        max_iter = 20
        iter = 0
        while True:
            iter += 1
            if(iter > max_iter):
                return self.initialize_state()
            b_king_idx = (np.random.randint(0, 4), np.random.randint(0, 4))
            # Occupied fields
            if b_king_idx != w_king_idx and b_king_idx != w_queen_idx:
                # King check
                if abs(b_king_idx[0] - w_king_idx[0]) > 1 and abs(b_king_idx[1] - w_king_idx[1]) > 1:
                    # Queen check straight
                    if b_king_idx[0] != w_queen_idx[0] and b_king_idx[1] != w_queen_idx[1]:
                        # Queen check diagonally ll <-> ur
                        if b_king_idx[0] - b_king_idx[1] != w_queen_idx[0] - w_queen_idx[1]:
                            # Queen check diagonally lr <-> ul
                            if b_king_idx[0] + b_king_idx[1] != w_queen_idx[0] + w_queen_idx[1]:
                                break
        self.b_king[b_king_idx] = 1

    def print(self):
        for i in range(4):
            print(i, end=" ")
            for j in range(4):
                if self.w_king[i,j] == 1:
                    print("K", end=" ")
                elif self.w_queen[i, j] == 1:
                    print("Q", end=" ")
                elif self.b_king[i, j] == 1:
                    print("B", end=" ")
                else:
                    print(".", end=" ")
            print()
        print("  0 1 2 3")

    def get_valid_fields_for_b(self):
        x_e, y_e = np.unravel_index(np.argmax(self.b_king), self.b_king.shape)
        x_k, y_k = np.unravel_index(np.argmax(self.w_king), self.w_king.shape)
        x_q, y_q = np.unravel_index(np.argmax(self.w_queen), self.w_queen.shape)
        valid_field = []
        for i in range(max(0, x_e - 1), min(4, x_e + 2)):
            for j in range(max(0, y_e - 1), min(4, y_e + 2)):
                if abs(i - x_k) <= 1 and abs(j - y_k) <= 1:  # Adjacent to king
                    continue
                if (i == x_q or j == y_q or i - j == x_q - y_q or i + j == x_q + y_q) and not (
                        i - x_q == (x_k - x_q) * 3 and j - y_q == (
                        y_k - y_q) * 3):  # attacked by Queen and not in shadow
                    continue
                valid_field.append((i, j))
        checked = (x_e == x_q or y_e == y_q or x_e - y_e == x_q - y_q or x_e + y_e == x_q + y_q) and not (x_e - x_q == (x_k - x_q) * 3 and y_e - y_q == (y_k - y_q) * 3)
        return valid_field, checked

    def game_status(self):
        valid_field, checked = self.get_valid_fields_for_b()
        nr_moves = np.zeros((8,))
        nr_moves[len(valid_field)] = 1
        checked_oh = np.zeros((2,))
        checked_oh[int(checked)] = 1
        return nr_moves, checked_oh

    def do_action(self, action):
        idx = np.argmax(action)
        if idx < 16:  # King Moves
            new_pos = np.reshape(action[:16], (4, 4))
            self.w_king = new_pos
        else:
            new_pos = np.reshape(action[16:], (4, 4))
            self.w_queen = new_pos
        nr_moves, checked = self.game_status()
        reward = 0.0
        if nr_moves[0] == 1 and checked[1] == 1:
            reward = 1.0
            self.done = True
        elif nr_moves[0] == 1:
            reward = -0.1
            self.done = True
        return reward


    def get_valid_actions(self):
        x_e, y_e = np.unravel_index(np.argmax(self.b_king), self.b_king.shape)
        # King
        k_valid_fields = np.zeros((4, 4))
        x_k, y_k = np.unravel_index(np.argmax(self.w_king), self.w_king.shape)
        k_valid_fields[max(0, x_k-1):min(4, x_k+2), max(0, y_k-1):min(4, y_k+2)] = 1

        k_valid_fields -= (self.w_king + self.w_queen + self.b_king)
        k_valid_fields[np.where(k_valid_fields < 0)] = 0

        k_valid_fields[max(0, x_e-1):min(4, x_e+2), max(0, y_e-1):min(4, y_e+2)] = 0


        # Queen
        q_valid_fields = np.zeros((4, 4))
        x_q, y_q = np.unravel_index(np.argmax(self.w_queen), self.w_queen.shape)
        q_valid_fields[x_q, :] = 1
        q_valid_fields[:, y_q] = 1
        for i in range(4):
            for j in range(4):
                # Queen check diagonally
                if i - j == x_q - y_q or i + j == x_q + y_q:
                    q_valid_fields[i, j] = 1
                if abs(i - x_e) <= 1 and abs(j - y_e) <= 1:
                    if abs(i - x_k) > 1 or abs(j - y_k) > 1:
                        q_valid_fields[i, j] = 0
        x_q_k = x_q - x_k
        y_q_k = y_q - y_k
        if x_q_k == 0 or y_q_k == 0 or abs(x_q_k) == abs(y_q_k):
            x_iter = x_k - np.sign(x_q_k)
            y_iter = y_k - np.sign(y_q_k)
            while x_iter < 4 and y_iter < 4 and x_iter >= 0 and y_iter >= 0:
                q_valid_fields[x_iter, y_iter] = 0
                x_iter -= np.sign(x_q_k)
                y_iter -= np.sign(y_q_k)
        q_valid_fields -= (self.w_king + self.w_queen + self.b_king)
        q_valid_fields[np.where(q_valid_fields < 0)] = 0

        stacked = np.hstack((k_valid_fields.flatten(), q_valid_fields.flatten()))
        return stacked.reshape((len(stacked), 1))

    def move_b(self):
        fields, _ = chess.get_valid_fields_for_b()
        move = random.choice(fields)
        self.b_king = np.zeros((4,4))
        self.b_king[move] = 1

    @property
    def state(self):
        nr_moves, checked_oh = self.game_status()
        stacked = numpy.hstack((self.w_king.flatten(), self.w_queen.flatten(), self.b_king.flatten(), nr_moves, checked_oh))
        return stacked.reshape((len(stacked), 1))

    def clone(self):
        new_board = Chess(False)
        new_board.b_king = np.copy(self.b_king)
        new_board.w_king = np.copy(self.w_king)
        new_board.w_queen = np.copy(self.w_queen)
        new_board.done = self.done
        return new_board


episodes = 10000
epsi = 0.4
gamma = 0.1
nr_steps = []
for episode in range(episodes):
    if episode % 100 == 0:
        print(episode)
    chess = Chess()
    Qvalues, H = network.forward(chess.state)
    Qvalues -= (1 - chess.get_valid_actions()) * 100000
    action = epsilon_greedy_policy(np.array([Qvalues]), epsi).T
    count = 0
    while True:
        count += 1
        chess_prime = chess.clone()
        reward = chess.do_action(action)
        Qvalues, H_prime = network.forward(chess.state)
        Qvalues -= (1 - chess.get_valid_actions()) * 100000
        next_action = epsilon_greedy_policy(np.array([Qvalues]), epsi).T
        Q_prime = Qvalues[np.argmax(next_action)]
        target = (reward + gamma * Q_prime) * action
        output = Qvalues * action
        network.descent(chess_prime.state, target, H, output)
        action = next_action

        if chess.done:
            nr_steps.append(count)
            #print(count)
            break
        #chess.print()
        chess.move_b()
        #chess.print()
    epsi *= 0.999

pyplot.plot(nr_steps)
pyplot.show()