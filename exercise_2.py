import numpy as np
from math import log2
from tqdm import tqdm
from typing import Tuple
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torchvision.datasets import MNIST
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from sklearn.linear_model import LogisticRegression


def main():
    # Load data
    train_set = MNIST('./data', train=True, download=True, transform=ToTensor())

    # Transform to flat vector w/ range 0, 1
    train_array = train_set.data.view(-1, 784).numpy() / 255
    train_label = train_set.targets.numpy()

    # Plot some digits
    plot_digits(train_array, train_label)

    # Kick off stump routine (High runtime! I decreased size of data)
    # Remove indices if you want the full run
    result = find_best_split(train_array[:250, 350:550], train_label[:250])
    print('Minimal Entropy:\t{}\nBest Cutoff:\t{}\nBest Feature:\t{}'
          .format(result[0], result[1], result[2]))

    # Train sklearn logistic regression
    clf_sklearn = LogisticRegression(multi_class='multinomial',
                                     solver='saga',
                                     max_iter=20,
                                     verbose=1,
                                     n_jobs=-1)
    clf_sklearn.fit(train_array, train_label)

    # Train PyTorch logistic regression
    clf_pytorch = train_torch(train_set)

    # Feel free to checkout weights
    clf_sklearn_weights = clf_sklearn.coef_
    clf_sklearn_biases = clf_sklearn.intercept_

    clf_pytorch_weights = clf_pytorch.fc.weight.data
    clf_pytorch_biases = clf_pytorch.fc.bias.data


def plot_digits(imgs: np.ndarray, labels: np.ndarray):
    """
    Plots first 9 digits supplied by 'img' and 'labels'
    :param imgs: Array w/ images.
    :param labels: Array with matching labels
    """
    plt.figure()
    for i, (img, label) in enumerate(zip(imgs, labels)):
        if i >= 9:
            break
        plt.subplot(3, 3, i + 1)
        plt.tight_layout()
        plt.imshow(img.reshape(28, 28), cmap='gray', interpolation='none')
        plt.title('Ground Truth: {}'.format(label))
        plt.xticks([])
        plt.yticks([])
    plt.show()


def find_best_split(features: np.ndarray, y: np.ndarray) -> Tuple[float, float, int]:
    """
    Iterate over all columns to find the best feature to split on, depending on the
    lowest entropy.
    :param features: Feature matrix with shape (samples, features)
    :param y: Target vector
    :return: Tuple containing minimal entropy, best cutoff value and best feature
    """
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


def find_split_point(feature: np.ndarray, y: np.ndarray) -> Tuple[float, int]:
    """
    Function to find the best split point for *one* feature
    :param y: Target vector for the given node
    :param feature: features for the given node
    :return: A Tuple with minimal entropy and best cut off value
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


def entropy_given_number_obs(n_k: int, n_not_k: int) -> float:
    """
    Calculates entropy based on the fractions of classes.
    :param n_k: Number of observations for target class
    :param n_not_k: Number of observations that are not target class
    :return: Entropy
    """
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
    :param y: Target vector for this node
    :return: The entropy of the given node and number of samples
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
    Calculate entropy based on a cut point.
    :param y: Target vector
    :param indicator: Indicator vector of logical values
    :return: Entropy for the given cut
    """
    n = len(y)
    mask = np.ones(n, bool)
    mask[indicator] = False

    entropy_left, num_values_left = entropy_node(y[indicator])
    entropy_right, num_values_right = entropy_node(y[mask])

    return (num_values_left / n * entropy_left) + (num_values_right / n * entropy_right)


def train_torch(data: Dataset) -> nn.Module:
    """
    Creates, trains and returns a model for Logistic Regression given
    the MNIST dataset provided by torch.
    :param data: MNIST Dataset loaded via torch
    :return: Logistic regression model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(data,
                        batch_size=24,
                        shuffle=True,
                        num_workers=4,
                        pin_memory=True)
    model = LogRegModel()
    model.to(device)
    optimizer = Adam(model.parameters(), lr=0.001)
    ce_loss = CrossEntropyLoss()

    for e in range(1, 10 + 1):
        losses = []
        correct = 0
        for batch_data, target in tqdm(loader):
            batch_data.to(device)
            target.to(device)

            prediction = model(batch_data.view(-1, 784))

            loss = ce_loss(prediction, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            correct += (torch.argmax(prediction, dim=1) == target).float().sum()
            losses.append(loss)

        accuracy = correct / len(data.targets)
        print('Ep. {0}\t; {1:.3f} accuracy; {2:.3f} loss'
              .format(e, accuracy, torch.mean(torch.stack(losses))))

    return model


class LogRegModel(nn.Module):
    """
    Logistic Regression in PyTorch.
    No Softmax layer is included as the result will be obtained by the argmax
    function and applying a probability mapping won't change the outcome.
    Also torch's crossentropy loss expects unnormalized score.
    """
    def __init__(self):
        super(LogRegModel, self).__init__()
        self.fc = nn.Linear(in_features=784, out_features=100)

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.fc(x)


if __name__ == '__main__':
    main()
