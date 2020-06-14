import cvxpy as cp, numpy as np, numpy.random as random

def L2Norm(vector):
    return np.sqrt(sumSquares(vector))
def sumSquares(vector):
    return np.sum(np.square(vector))


def generateUniformRandomMatrices(rows, columns, lo=-1, hi=1):
    return random.uniform(lo, hi, (rows, columns))



def rgg(num_vertices=5, lo=1, hi=10, density=0.5):
    if num_vertices <= 0:
        print("No. of vertices must be positive")
        return

    Weight = lo + (hi - lo + 1) * generateUniformRandomMatrices(num_vertices, num_vertices, lo=0, hi=1)
    Weight = 0.5 * (Weight + Weight.T)

    probMat = generateUniformRandomMatrices(num_vertices, num_vertices, lo=0, hi=1)
    Connectivity = probMat >= density
    Connectivity = np.triu(Connectivity, 1)
    Connectivity = Connectivity + Connectivity.T
    adjacencyMatrix = np.multiply(Connectivity, Weight)
    return adjacencyMatrix


def generateCompleteBipartite(k1, k2):
    numNodes = k1 + k2
    list_of_nodes = list(range(numNodes))
    np.random.shuffle(list_of_nodes)
    first_part = list_of_nodes[:k1]
    second_part = list_of_nodes[k1:]
    out = np.zeros((numNodes, numNodes))
    for f in first_part:
        for s in second_part:
            out[f, s] = 1
            out[s, f] = 1
    return out


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
    prob.solve(solver=solver)
    return X,prob.value


def getCutWeight(X,adjacencyMatrix,numNodes,method=0):
    M = np.linalg.cholesky(X)
    u = np.random.uniform(-1, 1, numNodes)
    u = u / L2Norm(u)
    labels = M.T @ u
    labels = (((labels >= 0) * 1) - 0.5) * 2
    cutWt = 0
    for i in range(numNodes):
        for j in range(i + 1, numNodes):
            if labels[i] != labels[j]:
                cutWt = cutWt + adjacencyMatrix[i, j]

    return cutWt


def runSolver(numNodes,method=0,k1=8,k2=12):
    if method==1:
        adjacencyMatrix = generateCompleteBipartite(k1, k2)
        # print("adjacencyMatrix=\n",adjacencyMatrix)
        numNodes=k1+k2
    else:
        adjacencyMatrix = rgg(numNodes, density=0.5)
    averageWt = np.sum(adjacencyMatrix) / ((numNodes ** 2 - numNodes))
    Laplacian = getLaplacian(adjacencyMatrix)
    X, opt = solveProblem(Laplacian, numNodes, "CVXOPT")
    rank = np.linalg.matrix_rank(X.value)
    # print("X=\n",X.value)

    cutwt = getCutWeight(X.value, adjacencyMatrix, numNodes)
    print("numNodes = ", numNodes, "averageWt = ", averageWt, "rank = ", rank, "cut weight = ", cutwt, "opt = ", opt)
    return numNodes,averageWt,rank,cutwt,opt


def runTheCode(nodesRange,filename,method,k1Range,k2Range):
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
    random.seed(25)
    nodesRange = range(2,101,2)
    filename = "five.1.txt"
    runTheCode(nodesRange,filename,0,0,0)

    k1Range = range(2,21,2)
    k2Range = range(2,21,2)
    filename = "five.2.txt"

    runTheCode(nodesRange, filename, 1, k1Range, k2Range)





