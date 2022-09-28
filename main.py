import numpy as np
import matplotlib.pyplot as plt
from utilities import *
from single_neural import SingleNeural
from multiple_neurals import MultipleNeurals
from sklearn.datasets import make_circles

if __name__ == "__main__":

    np.set_printoptions(formatter={'float_kind':"{:.2f}".format})

    # X_train, y_train, X_test, y_test = load_data()
    # singleNeural = SingleNeural(X_train, y_train, X_test, y_test)
    # singleNeural.show_train_set()
    # singleNeural.train()

    # X_train, y_train, X_test, y_test = load_data()
    # # X_train.shape  (1000, 64, 64)
    # # y_train.shape  (1000, 1)
    # # X_test.shape   (200, 64, 64)
    # # y_test.shape   (200, 1)

    X, y = make_circles(n_samples=100, noise=0.1, factor=0.3, random_state=0)
    X_test, y_test = make_circles(n_samples=100, noise=0.1, factor=0.3, random_state=1)

    X = X.T
    y = y.reshape((1, y.shape[0]))

    X_test = X_test.T
    y_test = y_test.reshape((1, y_test.shape[0]))

    plt.scatter(X[0, :], X[1, :], c=y, cmap='summer', edgecolors='k')
    plt.show()

    mulipleNeurals = MultipleNeurals(X, y, X_test, y_test, n1=2)
    mulipleNeurals.train(n_iter=10000)
    mulipleNeurals.show_train_performance()