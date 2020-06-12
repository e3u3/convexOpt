import numpy as np

def L1Norm(vector):
    return np.sum(np.abs(vector))

def LInfNorm(vector):
    return np.max(np.abs(vector))

