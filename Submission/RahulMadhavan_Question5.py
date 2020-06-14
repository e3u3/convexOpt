import cvxpy as cp, numpy as np, numpy.random as random
import randomMatrix,utilities


def getLaplacian(adjacencyMatrix):
    degree = np.sum(adjacencyMatrix, axis=0)
    degree = np.diag(degree)
    Laplacian = degree - adjacencyMatrix
    return Laplacian

def solveProblem(Laplacian,numNodes,solver):
    eye = np.ones(numNodes)
    X = cp.Variable((numNodes, numNodes), PSD=True)
    cost = (0.25) * cp.trace(Laplacian @ X)
    constr = [X >> 0, cp.diag(X) == eye]
    prob = cp.Problem(cp.Maximize(cost), constr)
    opt = prob.solve(solver=solver)
    # print("prob=",prob,"opt=",opt,"prob.value=",prob.value)
    return X,prob.value


def getCutWeight(X,adjacencyMatrix,numNodes,method=0):
    # print("X=", X)
    M = np.linalg.cholesky(X)
    # print("M=", M)
    u = np.random.uniform(-1, 1, numNodes)
    u = u / utilities.L2Norm(u)
    # print("u=", u)
    labels = M.T @ u
    # Shortcut for setting all positive to 1 and negative to -1
    labels = (((labels >= 0) * 1) - 0.5) * 2
    # print("labels=", labels)
    cutWt = 0
    for i in range(numNodes):
        for j in range(i + 1, numNodes):
            if labels[i] != labels[j]:
                cutWt = cutWt + adjacencyMatrix[i, j]

    # print("MaxCut = ",cutWt)
    return cutWt


def runSolver(numNodes,method=0,k1=8,k2=12):
    if method==1:
        adjacencyMatrix = randomMatrix.generateCompleteBipartite(k1, k2)
        # print("adjacencyMatrix=\n",adjacencyMatrix)
        numNodes=k1+k2
    else:
        adjacencyMatrix = randomMatrix.rgg(numNodes, density=0.5)
    averageWt = np.sum(adjacencyMatrix) / ((numNodes ** 2 - numNodes))
    Laplacian = getLaplacian(adjacencyMatrix)
    X, opt = solveProblem(Laplacian, numNodes, "CVXOPT")
    rank = np.linalg.matrix_rank(X.value)
    # print("X=\n",X.value)

    cutwt = getCutWeight(X.value, adjacencyMatrix, numNodes)
    print("numNodes = ", numNodes, "averageWt = ", averageWt, "rank = ", rank, "cut weight = ", cutwt, "opt = ", opt)
    return numNodes,averageWt,rank,cutwt,opt


def PartsRunner(nodesRange,filename,method,k1Range,k2Range):
    random.seed(8)
    if method==1:
        k1s = []
        k2s = []
    else:
        nodes = []
    opts = []
    averageWts = []
    ranks = []
    cutwts = []
    if method==1:
        for k1 in k1Range:
            for k2 in k2Range:
                numNodes, averageWt, rank, cutwt, opt = runSolver(k1+k2, method, k1, k2)
                k1s.append(k1)
                k2s.append(k2)
                averageWts.append(averageWt)
                ranks.append(rank)
                cutwts.append(cutwt)
                opts.append(opt)

    else:
        for numNodes in nodesRange:
            numNodes, averageWt, rank, cutwt,opt = runSolver(numNodes,method,k1=0,k2=0)
            nodes.append(numNodes)
            averageWts.append(averageWt)
            ranks.append(rank)
            cutwts.append(cutwt)
            opts.append(opt)

    f = open(filename,"w")
    if method==1:
        f.write("k1,k2,AverageWeight,Rank,CutWeight,Opt\n")
        i=0
        for k1 in k1Range:
            for k2 in k2Range:
                k1 = k1s[i]
                k2 = k2s[i]
                averageWt = averageWts[i]
                rank = ranks[i]
                cutwt = cutwts[i]
                opt = opts[i]
                f.write(str(k1) + "," + str(k2) + "," + str(averageWt) + "," + str(rank) + "," + str(cutwt) + "," + str(
                    opt) + "\n")
                i+=1
    else:
        f.write("numNodes,AverageWeight,Rank,CutWeight,Opt\n")
        i = 0
        for i in range(len(nodes)):
                numNodes = nodes[i]
                averageWt = averageWts[i]
                rank = ranks[i]
                cutwt = cutwts[i]
                opt = opts[i]
                f.write(str(numNodes) + "," + str(averageWt) + "," + str(rank) + "," + str(cutwt) + "," + str(opt) + "\n")
                i += 1
    f.close()





if __name__ == "__main__":
    # We follow the sequence provided by Prof Chirayu in his notes
    nodesRange = range(2,101,2)
    filename = "outputs/Q5.2.txt"
    PartsRunner(nodesRange,filename,0,0,0)

    k1Range = range(2,21,2)
    k2Range = range(2,21,2)
    filename = "outputs/Q5.3.txt"

    PartsRunner(nodesRange, filename, 1, k1Range, k2Range)





