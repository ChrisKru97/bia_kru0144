import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np
from functions import functions

selectedFunction = "rosenbrock"
lengthOfAxis = 40
elementsPerAxis = 1000

edge = lengthOfAxis / 2

# SOMA
pop_size = 20
PRT = 0.1
path_length = 3
step = 0.11
M_max = 100


def isOutOfBounds(x, y): return x < -edge or x > edge or y > edge or y < -edge


def countObjectiveValue(individual):
    value = functions[selectedFunction](*individual)
    return value


def generatePopulation():
    population = []
    for _ in range(pop_size):
        individual = np.random.rand(d) * lengthOfAxis - edge
        value = countObjectiveValue(individual)
        population.append([*individual, value])
    return np.array(population)


def findLeader(population):
    best = population[0]
    best_index = 0
    for i, x in enumerate(population[1:]):
        if(x[2] < best[2]):
            best = x
            best_index = i + 1
    return best_index, best[:2], best[2]


migrations = np.array([generatePopulation()])
population = np.copy(migrations[-1])

for _ in range(M_max):
    leader_index, leader, leader_value = findLeader(population)
    for i, x in enumerate(population):
        if(i == leader_index):
            continue
        t = step
        start = x[:2]
        start_value = x[2]
        best_position = x
        while t < path_length:
            rnd = np.random.rand(d)
            prt_vector = np.where(rnd < PRT, 1, 0)
            new_pos = start + (leader - start) * t * prt_vector
            if not isOutOfBounds(*new_pos):
                new_value = countObjectiveValue(new_pos)
                if(new_value < best_position[2]):
                    best_position = np.array([*new_pos, new_value])
            t = t + step
        population[i] = best_position
    migrations = np.concatenate(
        (migrations, np.array([population], copy=True)))


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
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='r')
    return update


def plot3d(values):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    update = generateUpdateFromValues(values, ax)
    a = anim.FuncAnimation(fig, update, frames=len(
        migrations) - 1, interval=500, repeat_delay=2000, repeat=True)
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


plotHeatMap(migrations)
plot3d(migrations)
