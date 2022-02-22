import pickle
import random

import numpy as np


def _epsilon():
    return 1e-10


def logistic(A):
    return 1. / (1. + np.exp(-A))


def relu(A):
    return np.maximum(0, A)


def softmax(Z):
    Y = np.exp(Z - np.max(Z))
    Y /= np.sum(Y, axis=0)
    return Y


def categorical_accuracy(Y, T):
    return np.sum(np.argmax(Y, axis=1) == np.argmax(T, axis=1)) / len(T)


def batch(X, T, B=None):
    assert len(X) == len(T)

    if B is None:
        B = len(T)

    indexes = list(range(len(T)))
    random.shuffle(indexes)

    index = 0
    while index + B <= len(T):
        yield X[indexes[index:index + B]], T[indexes[index:index + B]]
        index += B


def _gradient(X, T, Y, H, W2):
    G1 = (2. / len(X)) * np.dot(np.dot(W2.T, (Y - T)) * H * (1. - H), X.T)
    G2 = (2. / len(X)) * np.dot((Y - T), H.T)
    return G1, G2


class Network:
    W1, W2 = None, None

    def __init__(self, K, input_dim, output_dim, eta=0.02, mu=0, rho=0.9, rmsprop=False):
        # Xavier initialization
        self.W1 = np.random.randn(K + 1, input_dim) * 1.0 / np.sqrt(input_dim)
        self.W2 = np.random.randn(output_dim, K + 1) * 1.0 / np.sqrt(K + 1)
        self.eta = eta
        self.mu = mu
        self.rho = rho
        self.V1 = np.zeros(self.W1.shape)
        self.V2 = np.zeros(self.W2.shape)
        self.l1 = None
        self.l2 = None
        self.rmsprop = rmsprop

    def descent(self, X, T, H, Y):
        G1, G2 = _gradient(X, T, Y, H, self.W2)

        if self.rmsprop:
            self.V1 = self.rho * self.V1 + (1 - self.rho) * np.square(G1)
            self.V2 = self.rho * self.V2 + (1 - self.rho) * np.square(G2)

            self.l1 = (self.eta / (np.sqrt(self.V1) + _epsilon()))
            self.l2 = (self.eta / (np.sqrt(self.V2) + _epsilon()))

            self.W1 -= self.l1 * G1
            self.W2 -= self.l2 * G2
        else:
            self.W1 -= self.eta * G1
            self.W2 -= self.eta * G2

    def forward(self, X):
        H = relu(np.dot(self.W1, X))
        H[0, :] = 1.
        Z = np.dot(self.W2, H)
        Y = relu(Z)
        return Y, H


def pickle_network(network, file_name):
    with open(file_name, "wb") as f:
        pickle.dump(network, f)


def depickle(file_name):
    with open(file_name, "rb") as f:
        return pickle.load(f)
