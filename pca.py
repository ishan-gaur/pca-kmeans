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
    S, W = pca_torch(X)
    print(S)
    print(W)


test()
