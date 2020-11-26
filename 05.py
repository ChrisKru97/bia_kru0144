import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np
from functions import functions

selectedFunction = "griewank"
startingPoint = (-15, -15)
lengthOfAxis = 40
elementsPerAxis = 1000

edge = lengthOfAxis / 2

# Differential Evolution
maxGenerations = 50  # G_MAXIM
mutationConstant = 0.5  # F
crossoverRange = 0.5  # CR
populationCount = 20  # NP


def countObjectiveValue(individual):
    value = functions[selectedFunction](*individual)
    return value


def generatePopulation():
    population = []
    for i in range(populationCount):
        newCitizen = np.random.rand(2) * lengthOfAxis - edge
        population.append(newCitizen)
    return np.array(population)


generations = np.array([generatePopulation()])

for _ in range(maxGenerations):
    population = np.copy(generations[-1])
    new_population = np.copy(generations[-1])
    for i, x in enumerate(population):
        originalIndices = False
        while not originalIndices:
            r = (*np.trunc(np.random.rand(3) * populationCount).astype(int), i)
            originalIndices = len(np.unique(r)) == len(r)
        r1, r2, r3, _ = r
        v = (population[r1] - population[r2]) * \
            mutationConstant + population[r3]
        for j, param in enumerate(v):
            if param > edge:
                v[j] = edge
            elif param < -edge:
                v[j] = -edge
        u = np.zeros(d)
        j_rnd = np.random.randint(0, d)
        for j in range(d):
            if np.random.uniform() < crossoverRange or j == j_rnd:
                u[j] = v[j]
            else:
                u[j] = x[j]
        oldVectorValue = countObjectiveValue(x)
        selectedValue = countObjectiveValue(u)
        if selectedValue <= oldVectorValue:
            new_population[i] = u
    generations = np.concatenate((generations, np.array([new_population])))


X = np.arange(-edge, edge, lengthOfAxis/elementsPerAxis)
Y = np.arange(-edge, edge, lengthOfAxis/elementsPerAxis)
X, Y = np.meshgrid(X, Y)
Z = np.array(functions[selectedFunction](X, Y))


def generateUpdateFromValues(values, ax):
    def update(i):
        print('Showing {0}. generation'.format(i))
        ax.clear()
        ax.plot_surface(X, Y, Z, alpha=0.6)
        points = values[i]
        points_values = functions[selectedFunction](*np.transpose(points))
        ax.scatter(points[:, 0], points[:, 1], points_values, color='r')
    return update


def plot3d(values):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    update = generateUpdateFromValues(values, ax)
    a = anim.FuncAnimation(fig, update, frames=len(
        generations) - 1, interval=500, repeat_delay=2000, repeat=True)
    plt.show()


def generateHeatMapUpdateFromValues(values, ax):
    def update(i):
        print('Showing {0}. generation'.format(i))
        ax.clear()
        im = plt.imshow(Z, cmap='hot', interpolation='nearest',
                        extent=[-edge, edge, -edge, edge])
        points = values[i]
        plt.scatter(points[:, 0], points[:, 1], color='w')
    return update


def plotHeatMap(values):
    fig, ax = plt.subplots()
    im = plt.imshow(Z, cmap='hot', interpolation='nearest',
                    extent=[-edge, edge, -edge, edge])
    update = generateHeatMapUpdateFromValues(values, ax)
    a = anim.FuncAnimation(fig, update, frames=len(
        values) - 1, interval=500, repeat_delay=2000, repeat=True)
    plt.show()


plotHeatMap(generations)
plot3d(generations)
