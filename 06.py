import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np
from functions import functions

selectedFunction = "rosenbrock"
lengthOfAxis = 40
elementsPerAxis = 1000

edge = lengthOfAxis / 2

# Particle swarm optimization
M_max = 50
pop_size = 15
c1 = 2
c2 = 2
v_min = -2
v_max = 2
w_s = 0.9
w_e = 0.4


def get_w(i): return w_s * (w_s - w_e) * i / M_max


def putToBounds(individual):
    individual = np.where(individual < -edge, -edge, individual)
    individual = np.where(individual > edge, edge, individual)
    return individual


def countObjectiveValue(individual):
    value = functions[selectedFunction](*individual)
    return value


def generateSwarm():
    swarm = []
    for _ in range(pop_size):
        individual = np.random.rand(d) * lengthOfAxis - edge
        # could be random, but inertia_weight is 0 for first iteration which will cancel the initial velocity
        velocity = np.array([0, 0])
        value = countObjectiveValue(individual)

        # every individual in form:
        #  [0] => actual vector with its function value in last position
        #  [1] => actual velocity
        #  [2] => individual best position
        swarm.append(np.array([np.array([*individual, value]),
                               velocity, np.array([*individual, value])], dtype=object))
    return np.array(swarm)


def findGlobalBest(population):
    best = population[0][0]
    for x in population[1:]:
        if(x[0][2] < best[2]):
            best = x[0]
    return best


migrations = np.array([generateSwarm()])
swarm = np.copy(migrations[-1])
gBest = findGlobalBest(swarm)\

for m in range(M_max):
    inertia_weight = get_w(m)
    for i, x in enumerate(swarm):
        next_v = inertia_weight * x[1] + c1 * np.random.uniform() * (
            x[2][:2] - x[0][:2]) + c2 * np.random.uniform() * (gBest[:2] - x[0][:2])
        next_v = np.where(next_v < v_min, v_min, next_v)
        next_v = np.where(next_v > v_max, v_max, next_v)
        new_pos = putToBounds(x[0][:2] + next_v)
        new_pos_value = countObjectiveValue(new_pos)
        if(new_pos_value < x[2][2]):
            next_x = np.array([np.array([*new_pos, new_pos_value]),
                               next_v, np.array([*new_pos, new_pos_value])], dtype=object)
        else:
            next_x = np.array(
                [np.array([*new_pos, new_pos_value]), next_v, x[2]], dtype=object)
        if(new_pos_value < gBest[2]):
            gBest = np.array([*new_pos, new_pos_value])
        swarm[i] = next_x
    migrations = np.concatenate((migrations, np.array([swarm], copy=True)))


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


# forgetting the velocities and p_best
migrations = np.array(
    list(map(lambda x1: np.array(list(map(lambda x2: x2[0], x1))), migrations)))

plotHeatMap(migrations)
plot3d(migrations)
