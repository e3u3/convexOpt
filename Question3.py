import randomMatrix, utilities
import numpy as np, numpy.random as random, cvxpy as cp
from math import exp


def svm_gendata(Np, Nn, distance):
    Xp = np.array([[2, -1], [2, 1]]) / np.sqrt(2) @ np.random.randn(2, Np)
    Xp[0, :] = Xp[0, :] + distance
    Xp[1, :] = Xp[1, :] - distance
    yp = np.ones(Np)

    Xn = np.array([[2, -1], [2, 1]]) / np.sqrt(2) @ np.random.randn(2, Nn)
    Xn[0, :] = Xn[0, :] - distance
    Xn[1, :] = Xn[1, :] + distance

    yn = - np.ones(Nn)

    X = np.hstack((Xp, Xn))
    y = np.hstack((yp, yn))

    return X, y


def runforNum(numPos, NumNeg, distance=2.5, verboseRes=False):
    numPos = 50
    numNeg = 50
    distance = 3
    X, y = svm_gendata(numPos, numNeg, distance)
    m = numPos + numNeg
    one = np.ones(m)
    lambd = cp.Variable(m)
    constraints = [lambd >= 0, lambd.T @ y == 0]
    Y = np.diag(y)
    sigma = X.T @ X

    obj = cp.Maximize(lambd.T @ one + cp.quad_form(lambd, -Y @ sigma @ Y) / 2)
    prob = cp.Problem(obj, constraints)
    opt = prob.solve(solver="CVXOPT", verbose=verboseRes)
    if verboseRes:
        print("optimal value", opt)
        print("lambda values are ", lambd.value)
        objectiveVal = lambd.value.T @ one - 0.5 * lambd.value.T @ Y @ sigma @ Y @ lambd.value
        print("opt = ", opt, "obj = ", objectiveVal)
    return opt


# 3.2
def Part2(numPos, numNeg, sizeRange):
    runforNum(numPos, numNeg, True)
    opts = []
    f = open("Q3.1.txt", "w")
    f.write("numPoints,separation,optimal\n")
    for i in sizeRange:
        for j in range(1):
            min_distance = 2.5
            distance_step = 0.5
            distance = min_distance + j * distance_step
            opt = runforNum(i, i, distance, False)
            # print(i,",",opt)
            f.write(str(i) + "," + str(distance) + "," + str(opt) + "\n")
            opts.append(opt)
    # print(opts)
    f.close()


def compute_K(x1, x2, sigma):
    return exp((-np.linalg.norm(x1 - x2) ** 2) / (sigma * sigma))


def compose_K_sigma(X, sigma, m):
    K = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            K[i, j] = compute_K(X.T[i], X.T[j], sigma)
    return K


def getError(X, sigma, lambd, y, m):
    # First count the negs
    maxIterate = -10 ** 3
    for i in range(m):
        if y[i] == -1:
            sum1 = 0
            for j in range(m):
                sum1 += lambd.value[j] * y[j] * compute_K(X.T[i], X.T[j], sigma)
            if sum1 > maxIterate:
                maxIterate = sum1

    # First count the positive pts
    minIterate = 10 ** 3
    for i in range(m):
        if y[i] == 1:
            sum1 = 0
            for k in range(m):
                sum1 += lambd.value[k] * y[k] * compute_K(X.T[i], X.T[k], sigma)
            if sum1 < minIterate:
                minIterate = sum1

    # Set b as mid
    b = -(maxIterate + minIterate) / 2

    # We expect this to be 0 anyways
    errors = 0
    for i in range(m):
        finalsum = 0
        for j in range(m):
            finalsum += lambd.value[j] * y[j] * compute_K(X.T[i], X.T[j], sigma)
        pred = finalsum + b
        if np.sign(pred) != y[i]:
            errors += 1
    return errors


# 3.3
def Part3(numPos, numNeg, distance):
    X, y = svm_gendata(numPos, numNeg, distance)
    m = numPos + numNeg
    # sigmas = np.array([10**-2, 10**-1, 0.5, 10, 10**2])
    sigmas = np.logspace(-2, 3, 100)
    lambd = cp.Variable(m)
    one = np.ones(m)
    constraints = [lambd >= 0, lambd.T @ y == 0]
    train_errors = []
    lambda_values = []
    opts = []
    sigmasArr = []
    for i in range(len(sigmas)):
        sigma = sigmas[i]
        Sigma = compose_K_sigma(X, sigma, m)
        obj = cp.Maximize(lambd.T @ one + cp.quad_form(lambd, -Y @ Sigma @ Y) / 2)
        prob = cp.Problem(obj, constraints)
        opt = prob.solve()
        print("optimal value", opt)
        # print("lambda values are ", lambd.value)
        sigmasArr.append(sigma)
        opts.append(opt)
        lambda_values.append(lambd.value)
        tError = getError(X, sigma, lambd, y, m)

        train_errors.append(tError / m)
        print(train_errors[i])

    f = open("Q3.2.txt", "w")
    f.write("sigma,error,tError\n")
    for i in range(len(sigmasArr)):
        sigma = sigmasArr[i]
        opt = opts[i]
        tError = train_errors[i]
        f.write(str(sigma) + "," + str(opt) + "," + str(tError) + "\n")

    # print(opts)
    f.close()


if __name__ == "__main__":
    seed = 10
    numPos = 50
    numNeg = 50
    sizeRange = range(2, 101, 2)
    Part2(numPos, numNeg, sizeRange)
    Part3(numPos, numNeg, 2.5)
