import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np
import requests
import re

nodeCount = 20  # = ant count
tripCount = 50
axisLength = 100

# ANT colony params
alpha = 1  # degree of importance of pheromone
beta = 1  # degree of importance of distance
vap_coeff = 0.1
Q = 1


def generateLabels():
    return [chr(x) for x in range(97, 97 + nodeCount)]
    #r = requests.get('https://en.wikipedia.org/wiki/List_of_largest_cities')
    #m = np.array(re.findall(r'<tr>\n<td align="left"><a href="\/wiki\/.+" title=".+">(.+)<\/a>',r.text))
    # return m[:n]


def getDistance(a, b): return np.sqrt(np.sum((a - b) ** 2))


def countTotalDistance(distanceMatrix, path):
    distance = 0
    for i in range(nodeCount):
        distance += distanceMatrix[path[i], path[i + 1]]
    return distance


def generatePopulation(citiesList):
    population = []
    startingCity = citiesList[0]
    for _ in range(nodeCount):
        newCitizen = np.copy(citiesList[1:])
        np.random.shuffle(newCitizen)
        newCitizen = np.concatenate(
            ([startingCity], newCitizen, [startingCity]))
        population.append(newCitizen)
    return np.array(population)


def generateDistanceMatrix():
    nodePositions = np.random.rand(nodeCount, 2) * axisLength
    distanceMatrix = np.zeros((nodeCount, nodeCount))
    for i in range(nodeCount):
        for j in range(nodeCount):
            if(i != j and distanceMatrix[i][j] == 0):
                if(distanceMatrix[j][i] != 0):
                    distanceMatrix[i][j] = distanceMatrix[j][i]
                else:
                    distanceMatrix[i][j] = getDistance(
                        nodePositions[i], nodePositions[j])
    return distanceMatrix, nodePositions


labels = generateLabels()  # labels for nodes
distances, positions = generateDistanceMatrix()  # distance matrix
pheromones = np.ones((nodeCount, nodeCount))  # pheromone matrix
# array containing path from every iteration (choosing the first one, skipping search for the best one)
bestPathFromGeneration = []
pheromonesFromGeneration = []

for _ in range(tripCount):
    best_path_length = None
    best_path = None
    paths = []
    pathsLengths = []

    for i in range(nodeCount):  # ant searching path
        path = [i]  # every and starts from different node
        currentNode = i
        toBeVisited = list(range(nodeCount))
        del toBeVisited[i]  # cannot visit the same node again

        while len(path) != nodeCount:  # searching for next node to visit
            # counting probability of possible paths
            possibility = np.array(
                [(pheromones[currentNode, j] ** alpha) * ((1 / distances[currentNode, j]) ** beta) for j in toBeVisited])
            possibilitySum = np.sum(possibility)
            probability = possibility / possibilitySum
            cumulativeProbability = np.array(
                [np.sum(probability[0:j + 1]) for j in range(len(toBeVisited))])

            # selecting next node randomly
            random = np.random.rand()
            chosenPathIndex = np.argmin(np.abs(random - cumulativeProbability))

            # updating path list
            currentNode = toBeVisited[chosenPathIndex]
            path.append(currentNode)
            del toBeVisited[chosenPathIndex]

        path.append(i)
        lengthOfPath = countTotalDistance(distances, path)

        paths.append(path)
        pathsLengths.append(lengthOfPath)

        if i == 0 or lengthOfPath < best_path_length:
            best_path_length = lengthOfPath
            best_path = path

    paths = np.array(paths)

    # vaporization
    pheromones *= (1 - vap_coeff)

    # pheromones
    for i in range(nodeCount):
        for j in range(nodeCount):
            pheromonesToLeave = 0
            for k, path in enumerate(paths):
                i_index = np.where(path == i)[0][0]  # index of current node
                if path[i_index+1] == j:
                    pheromonesToLeave += Q / pathsLengths[k]
            pheromones[i, j] += pheromonesToLeave

    pheromonesFromGeneration.append(np.copy(pheromones))
    bestPathFromGeneration.append(np.array(best_path))

fig, ax = plt.subplots()


def update(i):
    ax.clear()
    current = bestPathFromGeneration[i]
    currentPheromones = pheromonesFromGeneration[i]
    lowestPheromone = currentPheromones.min()
    highestPheromone = currentPheromones.max()
    print('Showing best of {0}. iteration with length = {1}'.format(
        i, countTotalDistance(distances, current)))
    toPlot = [positions[x] for x in current]
    for index, point in enumerate(toPlot[:-1]):
        pheromoneIntensity = ((currentPheromones[current[index], current[index+1]] -
                               lowestPheromone) / (highestPheromone - lowestPheromone)) * 0.6 + 0.2
        ax.plot(*np.transpose((point, toPlot[index+1])),
                '-or', color=(pheromoneIntensity / 0.8, pheromoneIntensity, pheromoneIntensity))
    for i, txt in enumerate(current):
        ax.annotate(labels[txt], toPlot[i])


a = anim.FuncAnimation(fig, update, interval=500,
                       repeat_delay=2000, frames=tripCount, repeat=True)
plt.show()
