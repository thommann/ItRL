import numpy
import sklearn


def load_categorical_iris():
    return _load_data("categorical iris")


def load_categorical_digits():
    return _load_data("categorical digits")


def _load_data(dataset):
    X, T = {
        "categorical iris": (lambda: _load_categorical("iris")),
        "categorical digits": (lambda: _load_categorical("digits")),
    }[dataset]()

    print(F"Loaded data with {len(T)} samples of {X.shape[1]} dimensions.")
    return X, T


def _load_categorical(subj):
    if subj == "iris":
        dataset = sklearn.datasets.load_iris()
        O = 3
    elif subj == "digits":
        dataset = sklearn.datasets.load_digits()
        O = 10
    else:
        return

    X = dataset.data
    X = numpy.hstack((numpy.ones((len(X), 1)), X))

    T = numpy.zeros((len(X), O))
    for i, t in enumerate(dataset.target):
        T[i, t] = 1

    return X, T
