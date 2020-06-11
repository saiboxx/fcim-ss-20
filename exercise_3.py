import numpy as np
from math import log
from tqdm import tqdm
import matplotlib.pyplot as plt
from statistics import NormalDist
from scipy.stats import binom
from sklearn.linear_model import LogisticRegression


def main():
    np.random.seed(1337)
    print('~~~ Running Ex. 1 ~~~')
    exercise_1()
    print('~~~ Running Ex. 2 ~~~')
    exercise_2()
    print('~~~ Running Ex. 3 ~~~')
    exercise_3()


def exercise_1():
    x1 = np.random.uniform(-0.5, 0.5, 200)
    x2 = np.random.uniform(-0.5, 0.5, 200)
    y = np.where(np.sqrt(x1 ** 2 + x2 ** 2) + np.random.normal(0, 0.1, 200) < 0.3, 0, 1)

    X = np.vstack([x1, x2, x1 ** 2 + x2 ** 2]).transpose()

    clf = LogisticRegression(multi_class='multinomial',
                             solver='saga',
                             max_iter=100,
                             verbose=1,
                             n_jobs=-1)
    clf.fit(X, y)
    plot_decision_boundary(clf, X, y)


def exercise_2():
    # Draw from binomial dist. and calculate KLD to diverse normal dists.
    # -------------------------------------------------------------------------
    nr_points = 1000
    p = 0.5
    n = 100

    X = np.random.binomial(n, p, nr_points)
    plt.hist(X)
    plt.show()

    normal_distributions = [
        lambda x: NormalDist(mu=n * p, sigma=np.sqrt(n * p * (1 - p))).pdf(x),
        lambda x: NormalDist(mu=n * p - 10, sigma=np.sqrt(n * p * (1 - p))).pdf(x),
        lambda x: NormalDist(mu=n * p, sigma=2 * np.sqrt(n * p * (1 - p))).pdf(x),
        lambda x: NormalDist(mu=n * p + 20, sigma=0.5 * np.sqrt(n * p * (1 - p))).pdf(x)
    ]

    [plt.plot(range(n), [normal(x) for x in range(n)]) for normal in normal_distributions]
    plt.show()

    print('KL-Divergence optimal:\t{}'.format(kld_value(n * p, np.sqrt(n * p * (1 - p)), n, p)))
    print('KL-Divergence shift:\t{}'.format(kld_value(n * p - 10, np.sqrt(n * p * (1 - p)), n, p)))
    print('KL-Divergence scale increase:\t{}'.format(kld_value(n * p, 2 * np.sqrt(n*p*(1-p)), n, p)))
    print('KL-Divergence right scale increase:\t{}'.format(kld_value(n * p + 20, 0.5 * np.sqrt(n*p*(1-p)), n, p)))

    # Identify areas with high KLD given n and p.
    # -------------------------------------------------------------------------

    num_samples = 100
    xx, yy = np.mgrid[0.01:0.99:.01, 10:500:5]
    grid = np.c_[xx.ravel(), yy.ravel()]
    contour = [kld_value_approx(n=grid[i, 1], p=grid[i, 0], num_samples=num_samples)
               for i in tqdm(range(len(grid)), leave=False)]
    contour = np.asarray(contour).reshape(xx.shape)
    plt.contourf(xx, yy, contour)
    plt.show()


def exercise_3():
    X = sample_hypercube(2, 100)
    dist = norm_dist(X, 2)
    print('Point w/ minimum distance: {}'.format(X[np.argwhere(dist == min(dist))]))
    print('Point w/ maximum distance: {}'.format(X[np.argwhere(dist == max(dist))]))
    print('Min-Max Ratio: {}'.format(min_max_ratio(X, 2)))

    ratios = [min_max_ratio(sample_hypercube(3, 1000), 2) for _ in range(50)]
    plt.boxplot(ratios)
    plt.show()

    ratios_total = []
    dims = [5, 10, 100, 500, 1000, 5000]
    for d in dims:
        ratios = [min_max_ratio(sample_hypercube(d, 1000), 2) for _ in range(50)]
        ratios_total.append(ratios)

    plt.boxplot(ratios_total)
    plt.show()


def plot_decision_boundary(clf: LogisticRegression, x: np.ndarray, y: np.ndarray):
    xx, yy = np.mgrid[-0.6:0.6:.025, -0.6:0.6:.025]
    grid = np.c_[xx.ravel(), yy.ravel(), xx.ravel() ** 2 + yy.ravel() ** 2]
    y_hat = clf.predict(grid).reshape(xx.shape)
    y_hat_class = np.where(y_hat.squeeze() < 0.3, 0, 1)
    plt.title('Decision Boundary')
    plt.contourf(xx, yy, y_hat_class)
    plt.scatter(x[:, 0], x[:, 1], c=y)
    plt.show()


def kld_value(mu: float, sigma2: float, n: int, p: float) -> float:
    return 0.5 * log(sigma2) + 0.5 * sigma2 ** (-1) * (n * p * (1 - p) + (n * p - mu) ** 2)


def kld_value_approx(n: int, p: float, num_samples: int) -> float:
    # PMF Functions are not vectorized --> expect high runtime
    x = np.random.binomial(n, p, num_samples)
    log_binom = np.log([binom.pmf(x[i], n, p) for i in range(len(x))])
    log_normal = np.log([NormalDist(mu=n * p, sigma=np.sqrt(n * p * (1 - p))).pdf(x[i]) for i in range(len(x))])
    mean = np.mean(log_binom - log_normal)
    return mean if mean > 0 else 0


def sample_hypercube(dim: int, nsim: int) -> np.ndarray:
    return np.random.uniform(0, 1, (nsim, dim))


def norm_dist(x: np.ndarray, p: int) -> np.ndarray:
    return np.linalg.norm(x, p, axis=1)


def min_max_ratio(x: np.ndarray, p: int) -> float:
    dist = norm_dist(x, p)
    return (max(dist) - min(dist)) / min(dist)


if __name__ == '__main__':
    main()
