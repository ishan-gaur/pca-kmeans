import torch
import numpy as np
from pathlib import Path

def get_iris():
    data_path = Path('Data/iris.data')
    data = np.genfromtxt(data_path, delimiter=',', dtype='str')
    X = torch.from_numpy(data[:,:4].astype('float64'))
    labels = data[:,4]
    return X, labels
