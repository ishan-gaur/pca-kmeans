from pathlib import Path
import numpy as np
import random
import torch

def kmeans(k, X, epochs=1):
    """Performs Principle Component Analysis
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
    # print(centroids)
    # print(init_centr)
    # Calc distances for each vector
    dist = torch.empty(n, k)
    pred = torch.empty(n)
    # print(C)
    for _ in range(epochs):
        for j, c in enumerate(C):
            # print(c)
            dist[:,j] = torch.norm(torch.sub(X, c), dim=1)
        pred = torch.argmin(dist, 1)
        # for i in range(n): pred[i] = torch.argmin(X, 1)
        cts = torch.zeros(k) * 1e-20 # to prevent division by 0
        C = torch.zeros(k, d)
        for i, x in enumerate(X):
            cts[pred[i]] += 1
            C[pred[i]] += X[i]
        # print('pred:\n' + str(pred))
        C = torch.div(C, torch.reshape(cts, (-1, 1)))
        # print(C)
    return C, pred

def test():
    """Tests the PCA using supplied data file

    Todo:
        * Add support for different datafiles with argparse and specify range
        how many of the first n columns to use
        * Visualize results somehow as printed table, matplotlib, etc
    """
    data_path = Path('Data/iris.data')
    data = np.genfromtxt(data_path, delimiter=',', dtype='str')
    X = torch.from_numpy(data[:,:4].astype('float64'))
    labels = data[:,4]
    C, pred = kmeans(5, X)
    print(pred)
    C, pred = kmeans(3, X)
    print(pred)
test()
