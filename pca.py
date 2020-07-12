import timeit
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import torch

def pca_torch(X, center=True):
    """Performs Principle Component Analysis
    Args:
        X: torch tensor for the design matrix (containing example data)
        center: if True shifts distribution mean to 0

    Returns:
        (S, W): sv is a list of the singular values for each vector; W is the matrix,
        whose columns are the new basis (principle component vector)
    """
    X.sub_(torch.mean(X, 0))
    U, S, V = torch.svd(X)
    W = torch.t(V)
    return (S, W)

def test(pca=pca_torch):
    """Tests the PCA using supplied data file

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
    S, W = pca(X)
    print(S)
    print(W)

    # Time and generate histogram
    def timing_wrapper():
        pca(X)
    x = []
    for i in range(50000):
        x.append(timeit.timeit(timing_wrapper, number=1))
    x = np.asarray(x)
    print('Mean: ' + str(np.mean(x)) + '\tStdDev: ' + str(np.std(x)))
    plt.hist(x, bins=50)
    plt.show()

# test()
