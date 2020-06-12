import numpy as np, numpy.random as random, cvxpy as cp, matplotlib.pyplot as plt
import randomMatrix, utilities
import sys


def loss_fn(A, b, x):
    return cp.pnorm(A @ x - b, p=2) ** 2


def regularizer(x, norm=2, pow=2):
    return cp.pnorm(x, p=norm) ** pow


def objective_fn(A, b, x, lmbda, norm=2, pow=2):
    return loss_fn(A, b, x) + lmbda * regularizer(x, norm, pow)


def mse(X, Y, x):
    return (1.0 / X.shape[0]) * loss_fn(X, Y, x).value


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


def getIterationErrors(problem, tempFile, outFile):
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
        print(float(cols[4]))
        iterationErrors.append(float(cols[4]))
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
        iterationErrors = getIterationErrors(problem, tempFile, outFile)
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


if __name__ == "__main__":
    random.seed(8)
    m = 200
    n = 10
    A, b = randomMatrix.gendata_lasso(m, n)
    b = np.ndarray.flatten(b)
    print("A.shape=", A.shape)
    print("b.shape", b.shape)
    solveRegression(A, b, m, n, method=0, lmbda=0.1, showIterationErrors=1, tempFile="outputs/temp.txt",
                    outFile="outputs/iterationErrorMethod0.txt")
    solveRegression(A, b, m, n, method=1, lmbda=0.1, showIterationErrors=1, tempFile="outputs/temp.txt",
                    outFile="outputs/iterationErrorMethod1.txt")
    x = solveRegression(A, b, m, n, method=2, lmbda=0.1, showIterationErrors=1, tempFile="outputs/temp.txt",
                    outFile="outputs/iterationErrorMethod2.txt")

    print(x)

    x = cp.Variable(n)
    lmbda = cp.Parameter(nonneg=True)
    problem = cp.Problem(cp.Minimize(objective_fn(A, b, x, lmbda)))

    lmbda_values = range(5,100,5)
    train_errors = []
    test_errors = []
    x_values = []
    for v in lmbda_values:
        lmbda.value = v
        problem.solve()
        train_errors.append(mse(A, b, x))
        # test_errors.append(mse(X_test, Y_test, x))
        x_values.append(x.value)
    # %matplotlib inline
    # %config InlineBackend.figure_format = 'svg'
    plot_train_test_errors(train_errors, test_errors, lmbda_values)
    plot_regularization_path(lmbda_values, x_values)
