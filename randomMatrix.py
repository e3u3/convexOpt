import numpy as np
import numpy.random as random

def generateUniformRandomMatrices(rows,columns,lo=-1,hi=1):
    return random.uniform(lo,hi,(rows,columns))

def generateStdNormalRandomMatrices(rows,columns):
    return random.randn(rows,columns)

def sparseRandomNormalMatrix(rows,columns,density):
    A = generateStdNormalRandomMatrices(rows,columns)
    sparseA = A.copy()
    for i in range(rows):
        for j in range(columns):
            r = random.uniform(0,1)
            if(r>density):
                sparseA[i,j]=0
    return sparseA

def gendata_lasso(m=500,n=2500, noise=0, option=1):
    # function to generate test data for lasso
    #   Input:  m: no. of observations
    #           n: no. of features
    #       noise: standard deviation
    #      option: 0: no noise
    #              1: noise added by gaussian distribution
    #              2: noise added as an outlier (selecting any 1 of the
    #                 observations)
    ##
    x0 = sparseRandomNormalMatrix(n,1,0.05)
    A = generateStdNormalRandomMatrices(m, n)
    # normalize columns
    ANormalizer = np.square(A)
    ANormalizer = np.sum(ANormalizer,axis=0)
    ANormalizer = np.sqrt(ANormalizer)
    ANormalizer = 1/ANormalizer
    A = A.dot(np.diag(ANormalizer))

    v = np.sqrt(0.001) * generateStdNormalRandomMatrices(m, 1)
    b = A.dot(x0)  + v

    if option==1:
        b = b + noise * random.rand(b.shape[0],b.shape[1])
        return A,b

    if option == 2:
        randomRow = random.randint(m)
        b[randomRow] = b[randomRow] + noise * random.uniform(0,1)
        return A,b

    return A,b


    