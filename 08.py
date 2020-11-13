import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np
import requests
import re

nodeCount = 20 # = ant count
tripCount = 50
axisLength = 100

# ANT colony params
alpha = 2 # degree of importance of pheromone
beta = 1 # degree of importance of distance
vap_coeff = 0.1
Q = 2

def generateLabels():
    return [chr(x) for x in range(97, 97 + nodeCount)]
    #r = requests.get('https://en.wikipedia.org/wiki/List_of_largest_cities')
    #m = np.array(re.findall(r'<tr>\n<td align="left"><a href="\/wiki\/.+" title=".+">(.+)<\/a>',r.text))
    #return m[:n]

getDistance = lambda a, b: np.sqrt(np.sum((a - b) ** 2))

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
        newCitizen=np.concatenate(([startingCity],newCitizen,[startingCity]))
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
                    distanceMatrix[i][j] = getDistance(nodePositions[i], nodePositions[j])
    return distanceMatrix, nodePositions

labels = generateLabels() # labels for nodes
distances, positions = generateDistanceMatrix() # distance matrix
pheromones = np.ones((nodeCount, nodeCount)) # pheromone matrix
firstPathFromGeneration = [] # array containing path from every iteration (choosing the first one, skipping search for the best one)

for _ in range(tripCount):
    next_pheromones = np.copy(pheromones * (1 - vap_coeff)) # array for calculating pheromones for next iteration

    for i in range(nodeCount): # ant searching path
        path = [i] # every and starts from different node
        toBeVisited = list(range(nodeCount))
        del toBeVisited[i] # cannot visit the same node again

        while len(path) != nodeCount: # searching for next node to visit

            # counting probability of possible paths
            possibility = np.array([(pheromones[i, j] ** alpha) * ((1 / distances[i, j]) ** beta) for j in toBeVisited])
            possibilitySum = np.sum(possibility)
            probability = possibility / possibilitySum
            cumulativeProbability = np.array([np.sum(probability[0:i + 1]) for i in range(nodeCount - 1)])

            # selecting next node randomly
            random = np.random.uniform()
            chosenPathIndex = np.argmin(np.abs(random - cumulativeProbability))

            # updating path list
            path.append(toBeVisited[chosenPathIndex])
            del toBeVisited[chosenPathIndex]

        path.append(i)
        lengthOfPath = countTotalDistance(distances, path)

        if i == 0:
            firstPathFromGeneration.append(np.array(path)) # saving path

        # vaporization (used after full cycle)
        pheromoneToLeave = Q / lengthOfPath

        for j in range(nodeCount):
            next_pheromones[path[j + 1], path[j]] += pheromoneToLeave

    pheromones = next_pheromones

fig, ax = plt.subplots()
    
def update(i):
    ax.clear()
    current = firstPathFromGeneration[i]
    print('Showing {0}. iteration with length = {1}'.format(i, countTotalDistance(distances, current)))
    toPlot = np.array([positions[x] for x in current])
    ax.plot(*np.transpose(toPlot), '-or')
    for i, txt in enumerate(current):
        ax.annotate(labels[txt], toPlot[i])

a = anim.FuncAnimation(fig, update, interval=500, repeat_delay=2000, frames=tripCount, repeat=True)
plt.show()
