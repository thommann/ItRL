import random
import time

import numpy
import numpy as np


def _epsilon():
    return 1e-7


def logistic(A):
    return 1. / (1. + numpy.exp(-A))


def softmax(Z):
    Y = numpy.exp(Z - numpy.max(Z))
    Y /= numpy.sum(Y, axis=0)
    return Y


def categorical_accuracy(Y, T):
    return numpy.sum(numpy.argmax(Y, axis=1) == numpy.argmax(T, axis=1)) / len(T)


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





def _loss(X, T, W1, W2):
    Y, H, Z = _forward(X, W1, W2)
    J = - numpy.sum(numpy.sum(T * Y, axis=0) - numpy.log(numpy.sum(numpy.exp(Y), axis=0)))
    A = categorical_accuracy(Y.T, T.T)
    return [J, A], Y, H


def _gradient(X, T, Y, H, W2):
    G1 = (2. / len(X)) * numpy.dot(numpy.dot(W2.T, (Y - T)) * H * (1. - H), X.T)
    G2 = (2. / len(X)) * numpy.dot((Y - T), H.T)
    return G1, G2





def epsilon_greedy_policy(Qvalues, epsilon):
    N_class = numpy.shape(Qvalues)[1]
    batch_size = numpy.shape(Qvalues)[0]

    rand_values = numpy.random.uniform(0, 1, [batch_size])

    rand_a = rand_values < epsilon
    a = numpy.zeros([batch_size, N_class])

    for i in range(batch_size):

        if rand_a[i] == True:
            while 1:
                randi = numpy.random.randint(0, N_class)
                if Qvalues[i, randi] > -10000:
                    break
            a[i, randi] = 1

        else:

            a[i, numpy.argmax(Qvalues[i])] = 1

    return a


class Network:
    W1, W2 = None, None

    def __init__(self, K, input_dim, output_dim, eta=0.001, mu=0):
        # Xavier initialization
        self.W1 = numpy.random.randn(K + 1, input_dim) * 1.0 / numpy.sqrt(input_dim)
        self.W2 = numpy.random.randn(output_dim, K + 1) * 1.0 / numpy.sqrt(K + 1)
        self.eta = eta
        self.mu = mu


    def descent(self, X, T, H, Y):
        G1, G2 = _gradient(X, T, Y, H, self.W2)
        self.W1 -= self.eta * G1
        self.W2 -= self.eta * G2

    def forward(self, X):
        H = logistic(numpy.dot(self.W1, X))
        H[0, :] = 1.
        Z = numpy.dot(self.W2, H)
        Y = logistic(Z)
        return Y, H
