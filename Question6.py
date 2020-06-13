import numpy as np, cvxpy as cp, numpy.random as random

import matplotlib.pyplot as plt


def Solver(numVars, method=0):
    mOnes = np.ones(numVars)
    a = np.sort(random.uniform(-1, 1, numVars))

    p = cp.Variable(numVars)
    entropy = cp.sum(cp.entr(p))
    if method == 1:
        aSq = np.power(a, 2)
        aExp = 3 * np.power(a, 3) - 2 * a
        aLessPoint5 = [a < 0.5] * 1
        constraints = [p >= 0,
                       cp.matmul(mOnes, p) == 1,
                       p @ a <= 0.1,
                       p @ a >= -0.1,
                       p @ aSq >= 0.5,
                       p @ aSq <= 0.6,
                       p @ aExp >= -0.3,
                       p @ aExp <= -0.2,
                       p @ aLessPoint5 >= 0.3,
                       p @ aLessPoint5 <= 0.4]
    else:
        constraints = [p >= 0, cp.matmul(mOnes, p) == 1]

    prob = cp.Problem(cp.Maximize(entropy), constraints)
    prob.solve()
    print("numVars = ", numVars, "p.value=", p.value)
    return a, p.value


def PartsRunner(filename, numRange, method=0):
    f = open(filename, "w")
    f.write("numVariables,p values->\n")
    f.close()

    for numVars in numRange:
        f = open(filename, "a")
        a, pVals = Solver(numVars, method)

        if pVals is not None:
            f.write(str(numVars) + "-random-vals:" + ",")
            f.write(",".join(str(i) for i in a) + "\n")
            f.write(str(numVars) + "-probabilities:" + ",")
            f.write(",".join(str(i) for i in pVals) + "\n")
        else:
            f.write("Equations Not Satisfied for matrix size:"+str(numVars) + "\n")
        f.close()


if __name__ == "__main__":
    random.seed(8)
    numRange = range(2, 21, 2)
    filename = "outputs/pVals6.1.txt"
    PartsRunner(filename, numRange, 0)
    numRange = range(10, 31, 2)
    filename = "outputs/pVals6.2.txt"
    PartsRunner(filename, numRange, 1)
