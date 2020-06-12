import numpy as np

def L1Norm(vector):
    return np.sum(np.abs(vector))

def LInfNorm(vector):
    return np.max(np.abs(vector))

def sumSquares(vector):
    return np.sum(np.square(vector))

def L2Norm(vector):
    return np.sqrt(sumSquares(vector))
