import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt


def main():
    x, y, theta_1, theta_2 = generate_data(100)

    exercise_a(x, y)
    exercise_b()


def generate_data(n: int) -> Tuple[np.ndarray, np.ndarray, int, int]:
    x = np.random.normal(0, 1, (n, 2))
    noise = np.random.normal(0, 1, n)

    # True parameters
    theta_1 = 2
    theta_2 = 0

    y = x[:, 0] * theta_1 + x[:, 1] * theta_2 + noise

    # Normalize + center data
    x = (x - x.mean(axis=0)) / x.std(axis=0)
    y -= y.mean()

    return x, y, theta_1, theta_2


def exercise_a(x: np.ndarray, y: np.ndarray):
    theta_1 = np.arange(-3, 3, 0.05)
    theta_2 = np.arange(-3, 3, 0.05)

    # Visualize empirical risk
    T_1, T_2 = np.meshgrid(theta_1, theta_2)
    grid = np.c_[T_1.ravel(), T_2.ravel()]
    z = [r_emp(x, y, grid[i, 0], grid[i, 1]) for i in range(len(grid))]
    z = np.asarray(z).reshape(T_1.shape)
    plt.contour(T_1, T_2, z)

    # Perform gradient descent
    n = len(x)
    theta_old = [-2, -2]
    thetas_hist = [theta_old]
    steps = 50
    lr = 0.1

    for i in range(steps):
        theta_new_1 = theta_old[0] + lr * 2/n * np.matmul(x[:, 0].T, (y - x[:, 0] * theta_old[0] - x[:, 1] * theta_old[0]))
        theta_new_2 = theta_old[1] + lr * 2/n * np.matmul(x[:, 0].T, (y - x[:, 0] * theta_old[1] - x[:, 1] * theta_old[1]))
        thetas_hist.append([theta_new_1, theta_new_2])
        theta_old = [theta_new_1, theta_new_2]

    thetas_hist = np.array(thetas_hist)
    plt.plot(thetas_hist[:, 0], thetas_hist[:, 1])
    plt.scatter(thetas_hist[:, 0], thetas_hist[:, 1])
    plt.show()

    # Regularize the objective
    lam = 4
    z = [r_reg(x, y, grid[i, 0], grid[i, 1], lam) for i in range(len(grid))]
    z = np.asarray(z).reshape(T_1.shape)
    plt.contour(T_1, T_2, z)
    plt.show()

    # Show risk for changing theta_1
    y_theta = [r_reg(x, y, theta_1[i], -2, lam) for i in range(len(theta_1))]
    plt.plot(theta_1, y_theta)
    plt.show()


def r_emp(x: np.ndarray, y: np.ndarray, theta_1: int, theta_2: int) -> np.ndarray:
    return np.mean((y - x[:, 0] * theta_1 - x[:, 1] * theta_2)**2)


def r_reg(x: np.ndarray, y: np.ndarray, theta_1: int, theta_2: int, lam: int) -> np.ndarray:
    return np.mean((y - x[:, 0] * theta_1 - x[:, 1] * theta_2)**2) + lam * (abs(theta_1) + abs(theta_2))


def exercise_b():
    xx = np.arange(-2, 2, 0.01)
    yy = 2 * xx**2

    plt.plot(xx, yy)
    x_approx = np.arange(-2, 2, 0.1)
    lines_approx = [approx_fun(x_approx[i]) for i in range(len(x_approx))]
    [plt.plot(x_approx, x_approx * lines_approx[i][1] + lines_approx[i][0]) for i in range(len(x_approx))]
    plt.show()


def approx_fun(x_o: float) -> Tuple[float, float]:
    return -2 * x_o**2, 4 * x_o


if __name__ == '__main__':
    main()