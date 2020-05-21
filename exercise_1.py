import numpy as np
from time import time
from typing import Optional
from sklearn.linear_model import LinearRegression


def main():
    np.random.seed(42)
    n = 10000
    p = 100
    nr_sims = 100

    # Create data
    x = np.random.normal(size=(n, p))
    beta_truth = np.random.uniform(-2, 2, p)
    f_truth = x.dot(beta_truth)

    results = {
        'mse_sklearn': [],
        'mse_gd1': [],
        'mse_gd2': [],
        'time_sklearn': [],
        'time_gd1': [],
        'time_gd2': [],
    }

    for i, sim in enumerate(range(nr_sims)):
        print('Processing experiment {}.'.format(i))
        # Create response
        y = f_truth + np.random.normal(scale=2, size=n)

        start = time()
        coef_gd_1 = gradient_descent(stepsize=0.0001, x=x, y=y)
        delta = time() - start
        results['time_gd1'].append(delta)

        start = time()
        coef_gd_2 = gradient_descent(stepsize=0.00001, x=x, y=y)
        delta = time() - start
        results['time_gd2'].append(delta)

        start = time()
        lr = LinearRegression().fit(x, y)
        coef_lr = lr.coef_
        delta = time() - start
        results['time_sklearn'].append(delta)

        mse_sklearn = mse(coef_lr, beta_truth)
        mse_gd_1 = mse(coef_gd_1, beta_truth)
        mse_gd_2 = mse(coef_gd_2, beta_truth)
        results['mse_sklearn'].append(mse_sklearn)
        results['mse_gd1'].append(mse_gd_1)
        results['mse_gd2'].append(mse_gd_2)

    print('~~~~ AVERAGE LINEAR REGRESSION PERFORMANCE ~~~~')
    print('Num. experiments: {}\n'.format(nr_sims))
    print('Sklearn:\nProcessing Time: {0:.4f}ms\tCoef. MSE: {1:.6f}\n'
          .format(np.mean(results['time_sklearn']), np.mean(results['mse_sklearn'])))
    print('Gradient Descent 1:\nProcessing Time: {0:.4f}ms\tCoef. MSE: {1:.6f}\n'
          .format(np.mean(results['time_gd1']), np.mean(results['mse_gd1'])))
    print('Gradient Descent 2:\nProcessing Time: {0:.4f}ms\tCoef. MSE: {1:.6f}'
          .format(np.mean(results['time_gd2']), np.mean(results['mse_gd2'])))


def gradient_descent(stepsize: float,
                     x: np.ndarray,
                     y: np.ndarray,
                     beta: Optional[np.ndarray] = None,
                     eps: Optional[float] = 1e-8) -> np.ndarray:
    """
    Performs gradient descend for a linear regression model w/ L2 loss
    given the provided parameters. The procedure will be stopped if the
    change for a iteration is lass then eps.
    :param stepsize: The step_size in each iteration
    :param x: The step_size in each iteration
    :param y: The outcome vector y
    :param beta: The outcome vector y
    :param eps: A small constant measuring the changes in each update step
    :return: A set of optimal coefficients beta
    """
    if beta is None:
        beta = np.zeros(x.shape[1])
    change = float('inf')

    xtx = x.transpose().dot(x)
    xty = x.transpose().dot(y)

    while change > eps:
        delta = - stepsize * (xty - xtx.dot(beta))
        beta -= delta

        change = sum(abs(delta))

    return beta


def mse(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.mean((x - y)**2)


if __name__ == '__main__':
    main()
