import torch
import timeit
import random
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def kmeans_torch(k, X, epochs=100):
    """Performs K-means clustering
    Args:
        X: torch tensor for the design matrix (containing example data)
        center: if True shifts distribution mean to

    Returns:
        (C, pred): C are the new centroids; pred are the predicted clusters of
        the input data
    """
    # Pick random initial centroids
    n, d = X.size()
    if k > n:
        print('Number of centroids k, must be at most the number of examples in data\n' +
              'Error: k > n')
    init_centr = random.sample(range(0, X.size()[0]), k)
    C = torch.empty(k, d)
    for i, j in enumerate(init_centr):
        C[i] = X[j]
    # Training loop
    dist = torch.empty(n, k)
    pred = torch.empty(n)
    for _ in range(epochs):
        for j, c in enumerate(C):
            dist[:,j] = torch.norm(torch.sub(X, c), dim=1)
        pred = torch.argmin(dist, 1)
        cts = torch.zeros(k) * 1e-20 # to prevent division by 0
        C = torch.zeros(k, d)
        for i, x in enumerate(X):
            cts[pred[i]] += 1
            C[pred[i]] += X[i]
        C = torch.div(C, torch.reshape(cts, (-1, 1)))
    return C, pred

def test(kmeans=kmeans_torch):
    """Tests and times the kmeans using supplied data file

    Todo:
        * Add support for different datafiles with argparse and specify range
        how many of the first n columns to use
        * Visualize results somehow as printed table, matplotlib, etc
    """
    # Get data and run algorithm to demonstrate functionality
    data_path = Path('Data/iris.data')
    data = np.genfromtxt(data_path, delimiter=',', dtype='str')
    X = torch.from_numpy(data[:,:4].astype('float64'))
    labels = data[:,4]
    C, pred = kmeans(3, X)
    print(pred)

    # Time and generate histogram
    def timing_wrapper():
        kmeans(3, X)
    x = []
    for i in range(500):
        print(i)
        x.append(timeit.timeit(timing_wrapper, number=1))
    x = np.asarray(x)
    print(np.mean(x), np.std(x))
    plt.hist(x, bins=50)
    plt.show()

# test()
