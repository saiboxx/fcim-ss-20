from typing import Optional
from sklearn.datasets import make_blobs
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np


def main():
    np.random.seed(42)

    x, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=42)
    plt.scatter(x[:, 0], x[:, 1], c=y)
    plt.show()
    y = np.where(y == 0, -1, 1)
    x_ones = add_ones(x)

    mod_pegasos = pegasos_linear(y, x_ones, lamb=1/len(y))
    f_pegasos = np.matmul(x_ones, mod_pegasos)

    clf = np.sign(f_pegasos * y)
    accuracy = sum(np.where(clf > 0, 1, 0))/len(y)

    print('Accuracy: {}'.format(accuracy))
    print('Pegasos coefficents:\t{}'.format(mod_pegasos))

    mod_sklearn = LinearSVC(fit_intercept=False, max_iter=5000).fit(x_ones, y)
    print('Sklearn coefficents:\t{}'.format(mod_sklearn.coef_))


def pegasos_linear(
        y: np.ndarray,
        x: np.ndarray,
        nr_iter: Optional[int] = 50000,
        theta: Optional[np.ndarray] = None,
        lamb: Optional[float] = 1.0,
        alpha: Optional[float] = 0.01
) -> np.ndarray:
    """
    :param y: outcome vector
    :param x: design matrix (including a column of 1s for the intercept)
    :param nr_iter: number of iterations for the algorithm
    :param theta: starting values for thetas
    :param lamb: penalty parameter
    :param alpha: alpha step size for weight decay
    :return: weight vector
    """
    if theta is None:
        theta = np.random.normal(0, 1, x.shape[1])

    for _ in tqdm(range(nr_iter)):
        f_current = np.matmul(x, theta)

        sample = np.random.randint(len(x))

        theta *= (1 - lamb * alpha)
        if y[sample] * f_current[sample] < 1:
            theta += alpha * y[sample] * x[sample, :]

    return theta


def add_ones(x: np.ndarray) -> np.ndarray:
    """
    Adds intercept term to x
    :param x: data
    :return: data with intercept variable
    """
    return np.c_[np.ones(x.shape[0]), x]


if __name__ == '__main__':
    main()
