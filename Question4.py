import cvxpy as cp, numpy as np, numpy.random as random
from scipy import sparse
import matplotlib.pyplot as plt
import randomMatrix


def runSolver(mConstraint, L_Initial, S_Initial, m, n, lmbda=0.1):
    L = cp.Variable((m, n))
    S = cp.Variable((m, n))
    cost = cp.norm(L, "nuc") + lmbda * cp.norm(S, 1)
    constr = [L + S == mConstraint]
    prob = cp.Problem(cp.Minimize(cost), constr)
    prob.solve("MOSEK")
    lError = np.linalg.norm(L.value - L_Initial, 'fro')
    sError = np.linalg.norm(S.value - S_Initial, 'fro')
    return lError, sError


def generateLowRankMatrixPlusSparseMatrix(m, n, rank, density):
    lInitial = randomMatrix.generateLowRank(m, n, rank)
    sInitial = randomMatrix.sparseRandomNormalMatrix(m, n, density=density)
    M = lInitial + sInitial
    return M, lInitial, sInitial


def getErrors(mConstraint, L_Initial, S_Initial, m, n, lambda_vals):
    LErrors = []
    SErrors = []
    for lmbda in lambda_vals:
        print("Now running for lambda = ", lmbda)
        lError, sError = runSolver(mConstraint, L_Initial, S_Initial, m, n, lmbda)
        LErrors.append(lError)
        SErrors.append(sError)
    return LErrors, SErrors


if __name__ == "__main__":
    m = 50
    n = 50
    lambda_vals = np.logspace(-2, 3, 20)
    lambda_vals = range(1, 11)

    rank = 5
    density = 0.1
    mConstraint, L_Initial, S_Initial = generateLowRankMatrixPlusSparseMatrix(m, n, rank, density)
    LErrors, SErrors = getErrors(mConstraint, L_Initial, S_Initial, m, n, lambda_vals)
    filename = "outputs/4.2Errors.txt"
    f = open(filename, "w")
    f.write("Lambda,LErrors,SErrors\n")
    for i in range(len(LErrors)):
        lmbda = lambda_vals[i]
        lError = LErrors[i]
        sError = SErrors[i]
        f.write(str(lmbda) + "," + str(lError) + "," + str(sError) + "," + str(lError+sError) + "\n")
