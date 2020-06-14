import numpy as np, cvxpy as cp
from scipy.io import loadmat
from math import exp

def LoadData():
    imageTrain = loadmat('QuestionFiles/imageTrain.mat')['imageTrain'].reshape(784, 5000)
    labelTrain = np.squeeze(np.array(loadmat('QuestionFiles/labelTrain.mat')['labelTrain']))

    imageTest = loadmat('QuestionFiles/imageTest.mat')['imageTest'].reshape(784, 500)
    labelTest = np.squeeze(np.array(loadmat('QuestionFiles/labelTest.mat')['labelTest']))
    return imageTrain, labelTrain, imageTest, labelTest


def getDataForDigit(imageTrain, labelTrain, imageTest, labelTest, digit):
    DigitTrain = []
    for i in range(len(labelTrain)):
        if labelTrain[i] == digit:
            DigitTrain.append(imageTrain[:, i])

    lenTrain = len(DigitTrain)
    DigitTest = []
    for i in range(len(labelTest)):
        if labelTest[i] == digit:
            DigitTest.append(imageTest[:, i])

    lenTest = len(DigitTest)
    return DigitTrain, lenTrain, DigitTest, lenTest


def getTwoDigitsTrainTest(digit1, digit2, imageTrain, labelTrain, imageTest, labelTest):
    DigitTrain6, lenTrain6, DigitTest6, lenTest6 = getDataForDigit(imageTrain, labelTrain, imageTest, labelTest, digit1)

    DigitTrain8, lenTrain8, DigitTest8, lenTest8 = getDataForDigit(imageTrain, labelTrain, imageTest, labelTest, digit2)

    numOnenumTwoTrainLabel = np.hstack((-np.ones(lenTrain6), np.ones(lenTrain8)))
    numOnenumTwoTestLabel = np.hstack((-np.ones(lenTest6), np.ones(lenTest8)))
    numOnenumTwo_TrainImage = np.array(DigitTrain6 + DigitTrain8)
    numOnenumTwo_TestImage = np.array(DigitTest6 + DigitTest8)
    return numOnenumTwoTrainLabel, numOnenumTwoTestLabel, numOnenumTwo_TrainImage, numOnenumTwo_TestImage


def getAccuracyBetweenDigits(digit1, digit2, imageTrain, labelTrain, imageTest, labelTest):

    numOnenumTwoTrainLabel, numOnenumTwoTestLabel, numOnenumTwo_TrainImage, numOnenumTwo_TestImage = getTwoDigitsTrainTest(
        digit1, digit2, imageTrain, labelTrain, imageTest, labelTest)
    # print(numOnenumTwo_TrainImage.shape)
    # print(numOnenumTwo_TestImage.shape)

    n = numOnenumTwo_TrainImage.shape[1]
    W = cp.Variable((n))
    b = cp.Variable()
    ones = np.array(np.ones(numOnenumTwo_TrainImage.shape[0]))
    Y = np.diag(numOnenumTwoTrainLabel)

    obj = cp.Minimize((cp.pnorm(W, p=2) ** 2) / 2)
    constraints = [ones - Y @ (numOnenumTwo_TrainImage @ W - b * ones) <= 0]
    prob = cp.Problem(obj, constraints)
    prob.solve()

    W_final = W.value
    b_final = b.value
    # print(W_final.shape)
    errors = 0
    ones_test = ones = np.array(np.ones(numOnenumTwo_TestImage.shape[0]))
    # numOnenumTwoTestImage@W_final +b*ones
    pred = np.sign(numOnenumTwo_TestImage @ W_final + b_final * ones_test)
    # pred
    inAcc = (np.sign(numOnenumTwo_TestImage @ W_final + b_final * ones_test) != numOnenumTwoTestLabel).sum() / \
            numOnenumTwo_TestImage.shape[0]
    acc = 1 - inAcc
    return acc

def compute_K(x1, x2, sigma):
    return exp((-np.linalg.norm(x1 - x2) ** 2) / (sigma * sigma))


def compose_K_sigma(X, sigma,m):
    K = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            K[i, j] = compute_K(X.T[i], X.T[j], sigma)
    return K

def getAccuracyBetweenDigitsGaussianKerner(digit1, digit2, imageTrain, labelTrain, imageTest, labelTest):


    # Variable
    trainNum1Num2_label, testNum1Num2_label, trainNum1Num2_image, testNum1Num2_image = getTwoDigitsTrainTest(
        digit1, digit2, imageTrain, labelTrain, imageTest, labelTest)


    y = trainNum1Num2_label
    X = trainNum1Num2_image.T
    Y = np.diag(trainNum1Num2_label)
    m = y.shape[0]
    one = np.ones(m)
    lambd = cp.Variable(m)
    y_test = testNum1Num2_label
    X_test = testNum1Num2_image.T
    m_test = y_test.shape[0]


    constraints = [lambd >= 0, lambd.T @ y == 0]
    lambda_values = []
    sigma = 2.5
    Sigma = compose_K_sigma(X,sigma,m)
    obj = cp.Maximize(lambd.T @ one + cp.quad_form(lambd, -Y @ Sigma @ Y) / 2)
    prob = cp.Problem(obj, constraints)
    prob.solve()
    lambda_values.append(lambd.value)
    maxi = -999
    for j in range(m):
        if y[j] == -1:

            sum1 = 0
            for k in range(m):
                sum1 = sum1 + lambd.value[k] * y[k] * compute_K(np.squeeze(np.array(X.T[j])),
                                                                np.squeeze(np.array(X.T[k])), sigma)
            if sum1 > maxi:
                maxi = sum1

    mini = 1000
    for j in range(m):
        if y[j] == 1:

            sum1 = 0
            for k in range(m):
                sum1 = sum1 + lambd.value[k] * y[k] * compute_K(np.squeeze(np.array(X.T[j])),
                                                                np.squeeze(np.array(X.T[k])), sigma)
            if sum1 < mini:
                mini = sum1

    b = -(maxi + mini) / 2

    errors = 0
    for j in range(m_test):
        finalsum = 0
        for k in range(m):
            finalsum += lambd.value[k] * y[k] * compute_K(np.squeeze(np.array(X_test.T[j])),
                                                          np.squeeze(np.array(X.T[k])), sigma)
        pred = finalsum + b
        if np.sign(pred) != y_test[j]:
            errors += 1
    accuracy = 1 - (errors / m_test)
    return accuracy


if __name__ == "__main__":
    imageTrain, labelTrain, imageTest, labelTest = LoadData()
    print(labelTrain)
    # Making required Training Data
    accuracy = getAccuracyBetweenDigits(6, 8, imageTrain, labelTrain, imageTest, labelTest)
    print("accuracy =", accuracy)

    accuracies = []
    num1s = []
    num2s = []
    for i in range(1, 10, 1):
        for j in range(i + 1, 10, 1):
            accuracy = getAccuracyBetweenDigits(i, j, imageTrain, labelTrain, imageTest, labelTest)
            accuracies.append(accuracy)
            num1s.append(i)
            num2s.append(j)
    filename = "outputs/output3.4.txt"
    f = open(filename, "w")
    f.write("num1,num2,accuracy\n")
    for i in range(len(num1s)):
        num1 = num1s[i]
        num2 = num2s[i]
        accuracy = accuracies[i]
        f.write(str(num1) + "," + str(num2) + "," + str(accuracy) + "\n")
    f.close()

    accuracies = []
    num1s = []
    num2s = []
    for i in range(1, 10, 1):
        for j in range(i + 1, 10, 1):
            accuracy = getAccuracyBetweenDigitsGaussianKerner(i, j, imageTrain, labelTrain, imageTest, labelTest)
            print("num1=",i,"num1=",j,"accuracy=",accuracy)
            accuracies.append(accuracy)
            num1s.append(i)
            num2s.append(j)
    filename = "outputs/output3.4Gaussian.txt"
    f = open(filename, "w")
    f.write("num1,num2,accuracy\n")
    for i in range(len(num1s)):
        num1 = num1s[i]
        num2 = num2s[i]
        accuracy = accuracies[i]
        f.write(str(num1) + "," + str(num2) + "," + str(accuracy) + "\n")
    f.close()



