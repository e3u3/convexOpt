import numpy as np, numpy.random as random, cvxpy as cp
import randomMatrix, utilities
from cvxopt import matrix, solvers


# https://math.stackexchange.com/questions/1639716/how-can-l-1-norm-minimization-with-linear-equality-constraints-basis-pu
# We formulate the LP as minimize sum(t_i) for |A_ix - b_i| \leq t_i
# or in other words minimize sum(t_i) for (A_ix - b_i) \leq t_i and (A_ix - b_i) \geq -t_i
# Which can be written as minimize 1.T.dot(t) for (A_ix - b_i) \leq t_i and (A_ix - b_i) \geq -t_i
# OR minimize 0.T.dot(x) + 1.T.dot(t)
# such that
# (A_i x - t_i) \leq +b_i
# \forall i in [n]
# and
# -(A_ix + t_i) \leq -b_i
# and
# -t_i < 0
# \forall i in [n], where x \in R^r, t in R^n
#
# OR
# minimize 0.T.dot(x) + 1.T.dot(t)
# such that
# (A x - I t) \leq b
# and
# -Ax - I t \leq -b
# and
# 0x - I t \leq 0

def minimizeL1NormLP(A, b, n, d):
    opt_coeffs_x = np.zeros((d))
    opt_coeffs_t = np.ones((n))
    opt_coeffs = np.concatenate((opt_coeffs_x, opt_coeffs_t))
    opt_coeffs = matrix(opt_coeffs)
    # The third constraint is actually redundant
    # constraint_coeffs = np.vstack((np.hstack((A, -np.eye(n))),np.hstack((-A, -np.eye(n))),np.hstack((np.zeros(A.shape), -np.eye(n)))))
    # constraint_offsets = np.concatenate((-b,b,np.zeros(n)))

    constraint_coeffs = np.vstack((np.hstack((A, -np.eye(n))), np.hstack((-A, -np.eye(n)))))
    constraint_coeffs = matrix(constraint_coeffs)
    constraint_offsets = np.concatenate((b, -b))
    constraint_offsets = matrix(constraint_offsets)
    sol = solvers.lp(opt_coeffs, constraint_coeffs, constraint_offsets)
    out = sol['x'][:d, 0]
    return np.array(out)


# In the above formulation, instead of a vector t, we can just use a scalar t
# Then we write
# minimize t
# such that
# (A_ix - b_i) \leq t
# and
# (A_ix - b_i) \geq -t
# and
# t \geq 0
#
# The problem can be restated as
# minimize t
# such that
# (A_ix - t) \leq b_i
# and
# (-A_ix - t) \leq -b_i
# OR
# minimize 0.dot(x) + t
# such that
# (Ax - t1) \leq b
# and
# (-Ax - t1) \leq -b

def minimizeLInfNormLP(A, b, n, d):
    opt_coeffs_x = np.zeros((d))
    opt_coeffs_t = np.ones((1))
    opt_coeffs = np.concatenate((opt_coeffs_x, opt_coeffs_t))
    opt_coeffs = matrix(opt_coeffs)
    # The third constraint is actually redundant
    # constraint_coeffs = np.vstack((np.hstack((A, -np.eye(n))),np.hstack((-A, -np.eye(n))),np.hstack((np.zeros(A.shape), -np.eye(n)))))
    # constraint_offsets = np.concatenate((-b,b,np.zeros(n)))

    constraint_coeffs = np.vstack((np.hstack((A, -np.ones((n, 1)))), np.hstack((-A, -np.ones((n, 1))))))
    constraint_coeffs = matrix(constraint_coeffs)
    constraint_offsets = np.concatenate((b, -b))
    constraint_offsets = matrix(constraint_offsets)
    sol = solvers.lp(opt_coeffs, constraint_coeffs, constraint_offsets)
    out = sol['x'][:d, 0]
    return np.array(out)


def minimizeL1NormCVX(A, b, n, d):
    x_out = cp.Variable(d)
    # cvxL1Prob = cp.Problem(cp.Minimize(cp.norm(cp.matmul(A,x_out)-b,1)))
    cvxL1Prob = cp.Problem(cp.Minimize(cp.norm(A @ x_out - b, 1)))
    cvxL1Prob.solve("CVXOPT")
    # x = cp.Variable(d)
    # prob1 = cp.Problem(cp.Minimize(cp.norm(A.dot(x)-b,1)))
    return x_out.value


def minimizeLInfNormCVX(A, b, n, d):
    x_out = cp.Variable(d)
    # cvxL1Prob = cp.Problem(cp.Minimize(cp.norm(cp.matmul(A,x_out)-b,1)))
    cvxL1Prob = cp.Problem(cp.Minimize(cp.norm(A @ x_out - b, np.inf)))
    cvxL1Prob.solve("CVXOPT")
    # x = cp.Variable(d)
    # prob1 = cp.Problem(cp.Minimize(cp.norm(A.dot(x)-b,1)))
    return x_out.value


def fullprint(*args, **kwargs):
    from pprint import pprint
    import numpy
    opt = numpy.get_printoptions()
    numpy.set_printoptions(threshold=numpy.inf)
    pprint(*args, **kwargs)
    numpy.set_printoptions(**opt)


def printResults(A, b, x, typeMessage):
    d = A.shape[1]
    print("\n-------------------------------------------------------------------------------------------------------\n")
    print(typeMessage)
    print("\n-------------------------------------------------------------------------------------------------------\n")
    print("x=", np.ndarray.flatten(x))
    print("\n---------------------------\n")
    Adotxb1 = np.matmul(A, x.reshape(d, 1)) - b.reshape((n, 1))

    adotxprint = np.ndarray.flatten(Adotxb1)
    printOpt = np.get_printoptions()
    np.set_printoptions(threshold=d + 1)
    print("A.dot(x) - b =", adotxprint)
    np.set_printoptions(**printOpt)

    print("L1Norm(A.dot(x)-b) = ", utilities.L1Norm(Adotxb1))
    print("LInfNorm(A.dot(x)-b) = ", utilities.LInfNorm(Adotxb1))
    print("\n-------------------------------------------------------------------------------------------------------\n")


if __name__ == "__main__":
    random.seed(8)
    n = 200
    d = 10
    A = randomMatrix.generateUniformRandomMatrices(n, d)
    b = randomMatrix.generateUniformRandomMatrices(n, 1)[:, 0]
    x = minimizeL1NormLP(A, b, n, d)
    printResults(A, b, x, "Results for l1Norm though LP")
    x = minimizeL1NormCVX(A, b, n, d)
    printResults(A, b, x, "Results for l1Norm though CVXPY")

    x = minimizeLInfNormLP(A, b, n, d)
    printResults(A, b, x, "Results for lInfNorm though LP")

    x = minimizeLInfNormCVX(A, b, n, d)
    printResults(A, b, x, "Results for lInfNorm though CVXPY")
    print("Solutions for the two methods for both L-Inf norma nd L-1 Norm are exactly the same")
