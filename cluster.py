import torch
import itertools
import numpy as np
import matplotlib.pyplot as plt
from pca import pca_torch as PCA
from kmeans import kmeans_torch as Kmeans
from data_utils import get_iris

# PCA + Kmeans
def pca_kmeans(k, X, epochs=100):
    _, W = PCA(X)
    X = torch.matmul(X, W)
    return Kmeans(k, X)

def score(pred, labels):
    """Scores clustering predictions against true classes
    Args:
        pred: algorithm predictions (numerical classes)
        labels: true classes (must be numerical)

    Returns:
        score: max score over all permutations of predicted classes mapped to
        true classes. Note this means the clustering must produce the exact
        same number of classes as the expected classification.
    """
    scores = []
    classes = np.unique(labels)
    for perm in itertools.permutations(classes):
        perm = np.asarray(perm)
        f = lambda x: perm[x]
        pred_perm = f(pred)
        scores.append(np.sum((labels - pred_perm).astype(bool).astype(int)))
    # print(labels.shape)
    # print(scores)
    return (labels.shape[0] - min(scores)) / labels.shape[0]

def eval():
    X, labels = get_iris()
    # print(labels)
    _, labels = np.unique(labels, return_inverse=True)

    samples = 1000
    epochs = 100
    van_res = []
    pca_res = []
    for i in range(samples):
        _, van_pred = Kmeans(3, X, epochs)
        _, pca_pred = pca_kmeans(3, X, epochs)
        van_scr = score(van_pred.numpy(), labels)
        pca_scr = score(pca_pred.numpy(), labels)
        van_res.append(van_scr)
        pca_res.append(pca_scr)
        # print('Vanilla Kmeans: ' + str(van_scr))
        # print('With PCA: ' + str(pca_scr))
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2, sharex=ax1, sharey=ax1)
    ax1.hist(van_res, bins=20, rwidth=0.5)
    ax2.hist(pca_res, bins=20, rwidth=0.5)
    plt.show()

eval()
