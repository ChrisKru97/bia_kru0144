import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np
from functions import functions

selectedFunction = "rastrigin"
maxGuesses = 10000

d = 2

X = np.arange(-10, 10, 0.5)
Y = np.arange(-10, 10, 0.5)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

X, Y = np.meshgrid(X, Y)

Z = functions[selectedFunction](X, Y)

guesses = np.random.random((maxGuesses, X.ndim))*20 - 10

minValue = np.Inf

values = []

for x, y in guesses:
    value = functions[selectedFunction](x, y)
    if value < minValue:
        values.append([x, y, value])
        minValue = value


def update(i):
    ax.clear()
    ax.plot_surface(X, Y, Z, alpha=0.6)
    currentValues = values[i]
    ax.plot([currentValues[0]], [currentValues[1]], [currentValues[2]],
            markerfacecolor='r', markeredgecolor='r', marker='o', markersize=4)


a = anim.FuncAnimation(fig, update, frames=len(values) - 1, repeat=True)
plt.show()
