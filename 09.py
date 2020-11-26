import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np
from functions import functions

selectedFunction = "levy"
lengthOfAxis = 40
elementsPerAxis = 1000

edge = lengthOfAxis / 2

# Firefly
pop_size = 30  # fireflies count
i_max = 30  # iterations count
beta_0 = 1
alpha = 0.7

d = 2


def getToBounds(x): return tuple(edge if i > edge else -
                                 edge if i < -edge else i for i in x)


def countObjectiveValue(
    individual): return functions[selectedFunction](*individual)


def generatePopulation():
    pop = np.random.random((pop_size, d)) * lengthOfAxis - edge
    values = np.array([countObjectiveValue(x) for x in pop])
    return np.transpose(np.concatenate((np.transpose(pop), [values])))


iterations = np.array([generatePopulation()])
current_population = np.copy(iterations[-1])

for _ in range(i_max):
    intensities = 1 / current_population[:, d]
    best_firefly_index = intensities.argmax()
    for i in range(pop_size):
        i_position = current_population[i][:d]

        if i != best_firefly_index:
            for j in range(pop_size):
                if i != j and intensities[j] > intensities[i]:
                    j_position = current_population[j][:d]
                    distance = np.sqrt(
                        np.sum((i_position - j_position)**2))
                    i_position += (beta_0 / (1+distance)) * (
                        j_position - i_position)

        i_position += alpha * np.random.normal(0, 1, size=2)
        i_position = np.array(getToBounds(i_position))
        i_value = countObjectiveValue(i_position)

        intensities[i] = 1 / i_value
        current_population[i] = np.append(i_position, i_value)

    iterations = np.concatenate(
        (iterations, np.array([current_population], copy=True)))


X = np.arange(-edge, edge, lengthOfAxis/elementsPerAxis)
Y = np.arange(-edge, edge, lengthOfAxis/elementsPerAxis)
X, Y = np.meshgrid(X, Y)
Z = np.array(functions[selectedFunction](X, Y))


def generateUpdateFromValues(values, ax):
    def update(i):
        print('Showing {0}. generation'.format(i))
        points = values[i]
        intensities = 1 / points[:, 2]
        min_intensity = intensities.min()
        max_intensity = intensities.max()
        ax.clear()
        ax.plot_surface(X, Y, Z, alpha=0.6)
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=([(
            0.2 + 0.8 * ((intensity - min_intensity) / (max_intensity - min_intensity)), 0.5, 0.5) for intensity in intensities]))
    return update


def plot3d(values):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    update = generateUpdateFromValues(values, ax)
    a = anim.FuncAnimation(fig, update, frames=len(
        values) - 1, interval=500, repeat_delay=2000, repeat=True)
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


plotHeatMap(iterations)
plot3d(iterations)
