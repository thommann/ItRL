import random
from typing import Tuple

import numpy as np


class Chess:
    def __init__(self, initialize=True):
        self.w_king: Tuple[int, int] = (-1, -1)
        self.w_queen: Tuple[int, int] = (-1, -1)
        self.b_king: Tuple[int, int] = (-1, -1)
        self.done = False

        if initialize:
            self.initialize_state()

    def initialize_state(self):
        self.w_king = (np.random.randint(0, 4), np.random.randint(0, 4))

        while True:
            self.b_king = (np.random.randint(0, 4), np.random.randint(0, 4))
            if not self.w_king == self.b_king and not self.checked_by_king(self.b_king):
                break

        while True:
            self.w_queen = (np.random.randint(0, 4), np.random.randint(0, 4))
            if not self.w_king == self.w_queen and not self.b_king == self.w_queen and not self.checked_by_queen(
                    self.b_king):
                break

    def print(self):
        print("  0 1 2 3")
        for i in range(4):
            print(i, end=" ")
            for j in range(4):
                if self.w_king == (i, j):
                    print("K", end=" ")
                elif self.w_queen == (i, j):
                    print("Q", end=" ")
                elif self.b_king == (i, j):
                    print("E", end=" ")
                else:
                    print(".", end=" ")
            print()

    def get_valid_fields_for_b(self):
        x_e, y_e = self.b_king
        valid_fields = []
        for i in range(max(0, x_e - 1), min(4, x_e + 2)):
            for j in range(max(0, y_e - 1), min(4, y_e + 2)):
                field = (i, j)
                if not self.checked(field):
                    valid_fields.append(field)
        return valid_fields

    def checked(self, idx: Tuple[int, int]):
        return self.checked_by_king(idx) or self.checked_by_queen(idx)

    def checked_by_king(self, idx: Tuple[int, int]):
        x, y = idx
        x_k, y_k = self.w_king
        return abs(x - x_k) <= 1 and abs(y - y_k) <= 1

    def checked_by_queen(self, idx: Tuple[int, int]):
        x, y = idx
        x_k, y_k = self.w_king
        x_q, y_q = self.w_queen
        same_row = x == x_q
        same_column = y == y_q
        diagonal_ll_ur = x - y == x_q - y_q
        diagonal_ul_lr = x + y == x_q + y_q
        kings_shadow_1 = x - x_q == (x_k - x_q) * 3 and y - y_q == (y_k - y_q) * 3
        kings_shadow_2 = x - x_q == (x_k - x_q) * 2 and y - y_q == (y_k - y_q) * 2
        return (same_row or same_column or diagonal_ll_ur or diagonal_ul_lr) and not (kings_shadow_1 or kings_shadow_2)

    def game_status(self):
        # Enemy moves
        valid_fields = self.get_valid_fields_for_b()
        nr_moves = np.zeros((8,))
        nr_moves[len(valid_fields)] = 1
        # Check
        checked = self.checked(self.b_king)
        checked_oh = np.zeros((2,))
        checked_oh[int(checked)] = 1
        return nr_moves, checked_oh

    def do_action(self, action):
        idx = np.argmax(action)
        x = idx // 4
        y = idx % 4
        if idx < 16:
            # King move
            self.w_king = (x, y)
        else:
            # Queen move
            x -= 4
            self.w_queen = (x, y)

        nr_moves, checked = self.game_status()
        reward = -0.01
        if nr_moves[0] == 1 and checked[1] == 1:
            # Check
            reward = 1.0
            self.done = True
        elif nr_moves[0] == 1:
            # Stale
            reward = -1.0
            self.done = True
        return reward

    def get_valid_actions(self):
        x_e, y_e = self.b_king
        x_k, y_k = self.w_king
        x_q, y_q = self.w_queen

        # King
        k_valid_fields = np.zeros((4, 4))
        # Reachable
        k_valid_fields[max(0, x_k - 1):min(4, x_k + 2), max(0, y_k - 1):min(4, y_k + 2)] = 1
        # Enemy king
        k_valid_fields[max(0, x_e - 1):min(4, x_e + 2), max(0, y_e - 1):min(4, y_e + 2)] = 0
        # Occupied
        k_valid_fields[x_k, y_k] = 0
        k_valid_fields[x_q, y_q] = 0

        # Queen
        q_valid_fields = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                if self.checked_by_queen((i, j)) \
                        and not (abs(i - x_e) <= 1 and abs(j - y_e) <= 1 and (abs(i - x_k) > 1 or abs(j - y_k) > 1)):
                    # Reachable and protected
                    q_valid_fields[i, j] = 1

        # Occupied
        q_valid_fields[x_k, y_k] = 0
        q_valid_fields[x_q, y_q] = 0
        q_valid_fields[x_e, y_e] = 0

        stacked = np.hstack((k_valid_fields.flatten(), q_valid_fields.flatten()))
        return stacked.reshape((len(stacked), 1))

    def move_b(self):
        fields = self.get_valid_fields_for_b()
        self.b_king = random.choice(fields)

    @property
    def state(self):
        x_k, y_k = self.w_king
        w_king_oh = np.zeros((16,))
        w_king_oh[x_k * 4 + y_k] = 1

        x_q, y_q = self.w_queen
        w_queen_oh = np.zeros((16,))
        w_queen_oh[x_q * 4 + y_q] = 1

        x_e, y_e = self.b_king
        b_king_oh = np.zeros((16,))
        b_king_oh[x_e * 4 + y_e] = 1

        nr_moves_oh, checked_oh = self.game_status()

        stacked = np.hstack((w_king_oh, w_queen_oh, b_king_oh, nr_moves_oh, checked_oh))
        return stacked.reshape((len(stacked), 1))

    def clone(self):
        new_board = Chess(False)
        new_board.b_king = np.copy(self.b_king)
        new_board.w_king = np.copy(self.w_king)
        new_board.w_queen = np.copy(self.w_queen)
        new_board.done = self.done
        return new_board
