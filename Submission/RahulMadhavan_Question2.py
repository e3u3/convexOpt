import numpy as np, numpy.random as random, cvxpy as cp, matplotlib.pyplot as plt
import randomMatrix, utilities
import sys


def loss_fn(A, b, x):
    return cp.pnorm(A @ x - b, p=2) ** 2


def regularizer(x, norm=2, pow=2):
    return cp.pnorm(x, p=norm) ** pow


def objective_fn(A, b, x, lmbda, norm=2, pow=2):
    return loss_fn(A, b, x) + lmbda * regularizer(x, norm, pow)


def mse(A, b, x):
    return (1.0 / A.shape[0]) * loss_fn(A, b, x).value


def plot_regularization_path(lmbda_values, x_values):
    num_coeffs = len(x_values[0])
    for i in range(num_coeffs):
        plt.plot(lmbda_values, [wi[i] for wi in x_values])
    plt.xlabel(r"$\lambda$", fontsize=16)
    plt.xscale("log")
    plt.title("Regularization Path")
    plt.show()


def plot_train_test_errors(train_errors, test_errors, lmbda_values):
    plt.plot(lmbda_values, train_errors, label="Train error")
    # plt.plot(lmbda_values, test_errors, label="Test error")
    plt.xscale("log")
    plt.legend(loc="upper left")
    plt.xlabel(r"$\lambda$", fontsize=16)
    plt.title("Mean Squared Error (MSE)")
    plt.show()


def getIterationErrors(problem, tempFile, column):
    originalOut = sys.stdout
    sys.stdout = open(tempFile, "w")
    a = problem.solve(verbose=True)
    sys.stdout.close()
    sys.stdout = originalOut
    f = open(tempFile, "r")
    linenum = 0
    iterationErrors = []
    for line in f:
        linenum += 1
        if linenum < 5:
            continue
        if line == "\n":
            break
        cols = line.split("  ")

        iterationErrors.append(float(cols[column]))
    return iterationErrors


def solveRegression(A, b, m, n, method=0, lmbda=0.1, showIterationErrors=0, tempFile="outputs/temp.txt",
                    outFile="outputs/iterations.txt"):
    x = cp.Variable(n)

    if method == 0:
        problem = cp.Problem(cp.Minimize(objective_fn(A, b, x, 0)))
    elif method == 1:
        problem = cp.Problem(cp.Minimize(objective_fn(A, b, x, lmbda, norm=2, pow=2)))
    elif method == 2:
        problem = cp.Problem(cp.Minimize(objective_fn(A, b, x, lmbda, norm=1, pow=1)))
    else:
        problem = cp.Problem(cp.Minimize(objective_fn(A, b, x, 0)))

    if showIterationErrors:
        iterationErrors = getIterationErrors(problem, tempFile, 1)
        f = open(outFile, "w")
        f.write("IterationNumber,Errors\n")
        iterCounter = 1
        for iterError in iterationErrors:
            f.write(str(iterCounter) + "," + str(iterError) + '\n')
            iterCounter += 1
        f.close()

    else:
        problem.solve()
    return x.value


def Part1(A, b, m, n):
    solveRegression(A, b, m, n, method=0, lmbda=0.1, showIterationErrors=1, tempFile="outputs/temp.txt",
                    outFile="outputs/2.1iterationErrorMethod0.txt")
    solveRegression(A, b, m, n, method=1, lmbda=0.1, showIterationErrors=1, tempFile="outputs/temp.txt",
                    outFile="outputs/2.1iterationErrorMethod1.txt")
    solveRegression(A, b, m, n, method=2, lmbda=0.1, showIterationErrors=1, tempFile="outputs/temp.txt",
                    outFile="outputs/2.1iterationErrorMethod2.txt")


def LambdasCoordinatewiseWriteFile(xArray, lambdas, filename, multicoordinates=1):
    f = open(filename, "w")
    if multicoordinates:
        f.write("lambda," + ",".join(("Coordinate" + str(i + 1)) for i in range(len(xArray[0]))) + "\n")
    else:
        f.write("lambda,value" + "\n")

    for iter in range(len(xArray)):
        x = xArray[iter]
        lmbda = lambdas[iter]
        if multicoordinates:
            f.write(str(lmbda) + "," + ",".join(str(i) for i in x) + "\n")
        else:
            f.write(str(lmbda) + "," + str(x) + "\n")

    f.close()


def Part2(A, b, m, n, lambda_vals):
    method_xs = []
    lambdas = []

    for lmbda in lambda_vals:
        x = solveRegression(A, b, m, n, method=1, lmbda=lmbda, showIterationErrors=0)
        method_xs.append(x)
        lambdas.append(lmbda)
    methodFile = "outputs/2.2.1x.txt"
    LambdasCoordinatewiseWriteFile(method_xs, lambdas, methodFile, 1)

    AdotXMinusB = []
    L2Norms = []
    L2NormPlusRegs = []
    for i in range(len(method_xs)):
        x = method_xs[i]
        lmbda = lambdas[i]
        # normTwoSq = loss_fn(A, b, x)
        normTwoSq = utilities.L2Norm(A @ x - b.reshape(m, 1))
        regularizer = lmbda * (utilities.L2Norm(x))
        normTwoSqPlusReg = normTwoSq + regularizer
        AdotXMinusB.append(normTwoSq)
        L2Norms.append(regularizer)
        L2NormPlusRegs.append(normTwoSqPlusReg)
    methodFile = "outputs/2.2.1L2Adotxminusb.txt"
    LambdasCoordinatewiseWriteFile(AdotXMinusB, lambdas, methodFile, 0)

    methodFile = "outputs/2.2.1L2regularizer.txt"
    LambdasCoordinatewiseWriteFile(L2Norms, lambdas, methodFile, 0)

    methodFile = "outputs/2.2.1L2NormPlusRegularizer.txt"
    LambdasCoordinatewiseWriteFile(L2NormPlusRegs, lambdas, methodFile, 0)

    method_xs = []
    lambdas = []

    for lmbda in lambda_vals:
        x = solveRegression(A, b, m, n, method=2, lmbda=lmbda, showIterationErrors=0)
        method_xs.append(x)
        lambdas.append(lmbda)
    methodFile = "outputs/2.2.2x.txt"
    LambdasCoordinatewiseWriteFile(method_xs, lambdas, methodFile)

    AdotXMinusB = []
    L2Norms = []
    L2NormPlusRegs = []
    for i in range(len(method_xs)):
        x = method_xs[i]
        lmbda = lambdas[i]
        # normTwoSq = loss_fn(A, b, x)
        normTwoSq = utilities.L2Norm(A.dot(x) - b.reshape(m, 1))
        regularizer = lmbda * (utilities.L1Norm(x))
        normTwoSqPlusReg = normTwoSq + regularizer
        AdotXMinusB.append(normTwoSq)
        L2Norms.append(regularizer)
        L2NormPlusRegs.append(normTwoSqPlusReg)
    methodFile = "outputs/2.2.2L1Adotxminusb.txt"
    LambdasCoordinatewiseWriteFile(AdotXMinusB, lambdas, methodFile, 0)

    methodFile = "outputs/2.2.2L1regularizer.txt"
    LambdasCoordinatewiseWriteFile(L2Norms, lambdas, methodFile, 0)

    methodFile = "outputs/2.2.1L1NormPlusRegularizer.txt"
    LambdasCoordinatewiseWriteFile(L2NormPlusRegs, lambdas, methodFile, 0)


def Part3(m, n, lambda2, lambda3):
    sigma_min = 0.1
    sigma_step = 0.1
    method1Errors=[]
    method2Errors = []
    method3Errors = []
    sigmas = []
    for i in range(20):
        sigma = sigma_min + i * sigma_step
        sigmas.append(sigma)
        A, b = randomMatrix.gendata_lasso(m, n,sigma,1)
        b = np.ndarray.flatten(b)
        x = solveRegression(A, b, m, n, method=0)
        rmse_error = np.sqrt(mse(A, b, x))
        method1Errors.append(rmse_error)

        x = solveRegression(A, b, m, n, method=1, lmbda=lambda2)
        rmse_error = np.sqrt(mse(A, b, x))
        method2Errors.append(rmse_error)

        x = solveRegression(A, b, m, n, method=2, lmbda=lambda3)
        rmse_error = np.sqrt(mse(A, b, x))
        method3Errors.append(rmse_error)
    f = open("outputs/2.iii.txt","w")
    f.write("Sigma,RMSE_LSQ,RMSE_L2Norm,RMSE_L1Norm\n")
    for i in range(20):
        sigma = sigmas[i]
        m1e = method1Errors[i]
        m2e = method2Errors[i]
        m3e = method3Errors[i]
        f.write(str(sigma)+","+str(m1e)+","+str(m2e)+","+str(m3e)+"\n")

    f.close()

def Part4(m, n, lambda2, lambda3):
    sigma_min = 0.1
    sigma_step = 0.1
    method1Errors=[]
    method2Errors = []
    method3Errors = []
    sigmas = []
    for i in range(20):
        sigma = sigma_min + i * sigma_step
        sigmas.append(sigma)
        A, b = randomMatrix.gendata_lasso(m, n,sigma,2)
        b = np.ndarray.flatten(b)
        x = solveRegression(A, b, m, n, method=0)
        rmse_error = np.sqrt(mse(A, b, x))
        method1Errors.append(rmse_error)

        x = solveRegression(A, b, m, n, method=1, lmbda=lambda2)
        rmse_error = np.sqrt(mse(A, b, x))
        method2Errors.append(rmse_error)

        x = solveRegression(A, b, m, n, method=2, lmbda=lambda3)
        rmse_error = np.sqrt(mse(A, b, x))
        method3Errors.append(rmse_error)
    f = open("outputs/2.iv.txt","w")
    f.write("Sigma,RMSE_LSQ,RMSE_L2Norm,RMSE_L1Norm\n")
    for i in range(20):
        sigma = sigmas[i]
        m1e = method1Errors[i]
        m2e = method2Errors[i]
        m3e = method3Errors[i]
        f.write(str(sigma)+","+str(m1e)+","+str(m2e)+","+str(m3e)+"\n")

    f.close()





if __name__ == "__main__":
    random.seed(8)
    m = 200
    n = 10
    A, b = randomMatrix.gendata_lasso(m, n)
    b = np.ndarray.flatten(b)
    # print("A.shape=", A.shape)
    # print("b.shape", b.shape)

    Part1(A, b, m, n)
    lambda_vals = np.logspace(-2, 3, 50)
    # lambda_vals = range(5, 105, 5)
    Part2(A, b, m, n, lambda_vals)

    # For Part 3, we will use lambda = 2.25 for Part 2 and lambda = 0.1 for part 3

    Part3(200, 50, 2.25, 0.1)

    Part4(200, 50, 2.25, 0.1)

