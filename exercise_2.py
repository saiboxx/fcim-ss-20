from math import log2
from typing import Tuple
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST

def main():

    # Load data
    train_set = MNIST('./data', train=True, download=True)
    test_set = MNIST('./data', train=False, download=True)

    # Transform to flat vector w/ range 0, 1
    train_array = train_set.data.view(-1, 784).numpy() / 255
    test_array = train_set.data.view(-1, 784).numpy() / 255

    train_label = train_set.targets.numpy()
    test_label = train_set.targets.numpy()

    # Plot some digits
    #plot_digits(train_array[:9], train_label[:9])


    # Kick off stump routine (Large runtime! Decrease size of data)
    result = find_best_split(train_array, test_label)
    print('Minimal Entropy:\t{}\nBest Cutoff:\t{}\nBest Feature:\t{}'
          .format(result[0], result[1], result[2]))


def plot_digits(imgs: np.ndarray, labels: np.ndarray):
    plt.figure()
    for i, (img, label) in enumerate(zip(imgs, labels)):
        plt.subplot(3, 3, i + 1)
        plt.tight_layout()
        plt.imshow(img.reshape(28, 28), cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(label))
        plt.xticks([])
        plt.yticks([])
    plt.show()


def find_split_point(feature: np.ndarray, y: np.ndarray) -> Tuple[float, int]:
    """
    Function to find the best split point for *one* feature
    :param y: Set of outcome values in the node
    :param feature: One feature with the same number of
    rows as the length of set_of_values
    :return: A dict with minimal entropy and best cut off value
    """
    minimal_entropy = float('inf')
    unique_features = np.sort(np.unique(feature))
    best_cutoff = None

    for feat in unique_features:
        is_smaller_than_feat = feature < feat
        entropy_for_feat = entropy_overall(y, is_smaller_than_feat)

        if entropy_for_feat <= minimal_entropy:
            minimal_entropy = entropy_for_feat
            best_cutoff = feat

    return minimal_entropy, best_cutoff


def find_best_split(features: np.ndarray, y: np.ndarray) -> Tuple[float, float, int]:
    print('Iterating over all {} features'.format(features.shape[1]))
    overall_minimal_entropy = float('Inf')
    best_cutoff = best_feature = None

    for i in tqdm(range(features.shape[1])):
        min_entropy, cutoff = find_split_point(features[:, i], y)

        if min_entropy < overall_minimal_entropy:
            overall_minimal_entropy = min_entropy
            best_cutoff = cutoff
            best_feature = i

    return overall_minimal_entropy, best_cutoff, best_feature


def entropy_given_number_obs(n_k: int, n_not_k: int) -> float:
    n = n_k + n_not_k
    pi_k = n_k / n
    pi_not_k = n_not_k / n

    # Return in case of zero impurity
    if n_k == 0 or n_not_k == 0:
        return 0
    else:
        return - (pi_k * log2(pi_k) + pi_not_k * log2(pi_not_k))


def entropy_node(y: np.ndarray) -> Tuple[float, int]:
    """
    Function to get entropy for a node set
    :param y: Set of values in this node
    :return: The entropy of the set and number of values
    """
    n = len(y)
    entropy = 0.0
    classes = np.unique(y)

    for cls in classes:
        n_Nk = sum(y == cls)
        entropy += np.mean(y == cls) * entropy_given_number_obs(n_Nk, n - n_Nk)

    return entropy, n


def entropy_overall(y: np.ndarray, indicator: np.ndarray) -> float:
    """
    Calculate entropy based on a cut point
    :param y: Set of outcome values
    :param indicator: Indicator vector of logical values
    :return: Entropy
    """
    n = len(y)
    mask = np.ones(n, bool)
    mask[indicator] = False

    entropy_left, num_values_left = entropy_node(y[indicator])
    entropy_right, num_values_right = entropy_node(y[mask])

    return (num_values_left / n * entropy_left) + (num_values_right / n * entropy_right)


if __name__ == '__main__':
    main()