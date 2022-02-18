from Assignment import data
from Assignment.Network import Network

from matplotlib import pyplot

X, T = data.load_categorical_digits()

# Parameters
B = 100
K = 10
eta = 0.01
epochs = 10000
mu = 0.5

network = Network()

loss = network.learn(X, T, K, eta=eta, epochs=epochs, B=B, mu=mu)

pyplot.figure()

ax1 = pyplot.gca()
ax2 = ax1.twinx()
ax1.set_xlabel("Epoch")

ax1.loglog(loss[:, 0], color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")
ax1.set_ylabel("Loss", color="tab:blue")

ax2.semilogx(loss[:, 1], color="tab:red")
ax2.tick_params(axis="y", labelcolor="tab:red")
ax2.set_ylabel("Accuracy", color="tab:red")

pyplot.show()
