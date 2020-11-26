import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np
from functions import functions

selectedFunction = "ackley"
iterations = 10
pointsPerIteration = 100
lengthOfAxis = 20
sigma = lengthOfAxis/20
startingPoint = (9, 9)


def levy(x, y):
    w = [1+(x-1)/4, 1+(y-1)/4]
    return np.sin(np.pi*w[0])**2 + (w[0]-1)**2 * (1+10*np.sin(np.pi*w[0]+1)**2) + (w[1]-1)**2 * (1+10*np.sin(np.pi*w[1]+1)**2) + (w[1]-1)**2 * (1+np.sin(2*np.pi*w[1])**2)


X = np.arange(-lengthOfAxis/2, lengthOfAxis/2, lengthOfAxis/50)
Y = np.arange(-lengthOfAxis/2, lengthOfAxis/2, lengthOfAxis/50)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

X, Y = np.meshgrid(X, Y)

Z = functions[selectedFunction](X, Y)


def generateBlindSearchSteps():
    guesses = np.random.uniform(
        size=(iterations*pointsPerIteration, X.ndim))*20 - 10
    minValue = np.Inf
    values = []
    for x, y in guesses:
        value = functions[selectedFunction](x, y)
        if value < minValue:
            values.append([x, y, value])
            minValue = value
    return values


def generateHillClimbSteps():
    minPoints = []
    lastMinPoint = [startingPoint[0], startingPoint[1],
                    functions[selectedFunction](startingPoint[0], startingPoint[1])]
    for i in range(iterations):
        guesses = np.random.normal(
            (lastMinPoint[0], lastMinPoint[1]), sigma, (pointsPerIteration, X.ndim))
        minPoint = lastMinPoint
        for x, y in guesses:
            # Checking for going out of boundaries of rendered animation, can be deleted, but will the fluidity of animation
            if x > lengthOfAxis/2 or x < -lengthOfAxis/2 or y > lengthOfAxis/2 or x < -lengthOfAxis/2:
                continue
            value = functions[selectedFunction](x, y)
            if value < minPoint[2]:
                minPoint = [x, y, value]
        if minPoint[2] < lastMinPoint[2]:
            minPoints.append(minPoint)
            lastMinPoint = minPoint
    return minPoints


values = generateHillClimbSteps()


def update(i):
    ax.clear()
    ax.plot_surface(X, Y, Z, alpha=0.6)
    currentValues = values[i]
    ax.plot([currentValues[0]], [currentValues[1]], [currentValues[2]],
            markerfacecolor='r', markeredgecolor='r', marker='o', markersize=4)


a = anim.FuncAnimation(fig, update, frames=len(values) - 1, repeat=True)
plt.show()
