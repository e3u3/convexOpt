import numpy as np
import numpy.random as random

def generateUniformRandomMatrices(rows,columns,lo=-1,hi=1):
    return random.uniform(lo,hi,(rows,columns))



def gendata_lasso(m,n):
    ## function to generate test data for lasso
    #   Input:  m: no. of observations
    #           n: no. of features
    #       noise: standard deviation
    #      option: 0: no noise
    #              1: noise added by gaussian distribution
    #              2: noise added as an outlier (selecting any 1 of the
    #                 observations)
    ##

    