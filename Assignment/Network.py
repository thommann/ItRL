import random
import time

import numpy


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


def _forward(X, W1, W2):
    H = logistic(numpy.dot(W1, X))
    H[0, :] = 1.
    Z = numpy.dot(W2, H)
    Y = logistic(Z)
    return Y, H, Z


def _loss(X, T, W1, W2):
    Y, H, Z = _forward(X, W1, W2)
    J = - numpy.sum(numpy.sum(T * Y, axis=0) - numpy.log(numpy.sum(numpy.exp(Y), axis=0)))
    A = categorical_accuracy(Y.T, T.T)
    return [J, A], Y, H


def _gradient(X, T, Y, H, W2):
    G1 = (2. / len(X)) * numpy.dot(numpy.dot(W2.T, (Y - T)) * H * (1. - H), X.T)
    G2 = (2. / len(X)) * numpy.dot((Y - T), H.T)
    return G1, G2


def _descent(X, T, W1, W2, eta, W1_prev, W2_prev, mu, loss):
    J, Y, H = loss(X, T, W1, W2)
    G1, G2 = _gradient(X, T, Y, H, W2)
    W1 -= (1 - mu) * eta * G1 - mu * (W1 - W1_prev)
    W2 -= (1 - mu) * eta * G2 - mu * (W2 - W2_prev)
    return J


def _gradient_descent(X, T, W1, W2, W1_prev, W2_prev, B, eta, epochs, mu, loss):
    print(F"Performing Gradient Descent for {epochs} epochs.")
    losses = []
    start = time.time()
    for epoch in range(epochs):
        for x, t in batch(X, T, B):
            W1_copy = W1.copy()
            W2_copy = W2.copy()

            _descent(x.T, t.T, W1, W2, eta, W1_prev, W2_prev, mu, loss)

            W1_prev = W1_copy
            W2_prev = W2_copy

        J, Y, _ = loss(X.T, T.T, W1, W2)
        losses.append(J)
        if len(J) > 1:
            print(F"\repoch: {epoch} - loss: {J[0]} - accuracy: {J[1]} - time: {time.time() - start}", end="",
                  flush=True)
        else:
            print(F"\repoch: {epoch} - loss: {J[0]} - time: {time.time() - start}", end="", flush=True)

    print()
    return numpy.array(losses)


def epsilon_greedy_policy(Qvalues, epsilon):
    N_class = numpy.shape(Qvalues)[1]
    batch_size = numpy.shape(Qvalues)[0]

    rand_values = numpy.random.uniform(0, 1, [batch_size])

    rand_a = rand_values < epsilon
    a = numpy.zeros([batch_size, N_class])

    for i in range(batch_size):

        if rand_a[i] == True:

            a[i, numpy.random.randint(0, N_class)] = 1

        else:

            a[i, numpy.argmax(Qvalues[i])] = 1

    return a


class Network:
    W1, W2 = None, None

    def __init__(self):
        pass

    # X: input
    # T: target
    # K: hidden layer size
    # eta: learning rate
    # epochs: nr. epochs
    # B: batch size
    # mu: momentum rate
    def learn(self, X, T, K, eta=0.001, epochs=10000, B=None, mu=0):
        # Xavier initialization
        self.W1 = numpy.random.randn(K + 1, X.shape[1]) * 1.0/numpy.sqrt(X.shape[1])
        self.W2 = numpy.random.randn(T.shape[1], K + 1) * 1.0/numpy.sqrt(K + 1)

        W1_prev = self.W1
        W2_prev = self.W2

        return _gradient_descent(X, T, self.W1, self.W2, W1_prev, W2_prev, B, eta, epochs, mu, _loss)

    def predict(self, X):
        return _forward(X.T, self.W1, self.W2)[0].T
