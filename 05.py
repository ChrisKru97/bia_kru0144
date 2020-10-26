import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np

selectedFunction = "griewank"
startingPoint = (-15, -15)
lengthOfAxis = 40
elementsPerAxis = 1000

edge = lengthOfAxis / 2

# Differential Evolution
maxGenerations = 50 # G_MAXIM
mutationConstant = 0.5 # F
crossoverRange = 0.5 # CR
populationCount = 20 # NP

# Function variables
d = 2
m = 10
a = 20
b = 0.2
c = 2 * np.pi

def levy(x,y):
    w = [1+(x-1)/4,1+(y-1)/4]
    return np.sin(np.pi*w[0])**2 + (w[0]-1)**2 * (1+10*np.sin(np.pi*w[0]+1)**2) + (w[1]-1)**2 * (1+10*np.sin(np.pi*w[1]+1)**2) + (w[1]-1)**2 * (1+np.sin(2*np.pi*w[1])**2)

myFunctions = {
        "sphere" : lambda x, y : x*x + y*y,
        "schwefel": lambda x, y: 418.9829*d - (x*np.sin(np.sqrt(np.absolute(x)))+y*np.sin(np.sqrt(np.absolute(y)))),
        "rosenbrock": lambda x,y: (100*(y-x*x)**2+(x-1)**2),
        "rastrigin": lambda x,y: 10*d + (x*x - 10*np.cos(2*np.pi*x))+(y*y - 10*np.cos(2*np.pi*y)),
        "griewank": lambda x,y: ((x*x)/4000+(y*y)/4000)-((np.cos(x))*(np.cos(y/np.sqrt(2)))+1),
        "levy": levy,
        "michalewicz": lambda x,y: -(np.sin(x)*np.sin((x*x)/np.pi)**(2*m))-(np.sin(y)*np.sin((2*y*y)/(np.pi))**(2*m)),
        "zakharov": lambda x,y: x*x + y*y + (0.5*x + y)**2 + (0.5*x+y)**4,
        "ackley": lambda x,y: -a*np.exp(-b*np.sqrt((x*x+y*y)/2))-np.exp((np.cos(c*x)+np.cos(c*y))/2)+a+np.exp(1),
}

def countObjectiveValue(individual):
    value = myFunctions[selectedFunction](*individual)
    return value 

def generatePopulation():
    population = []
    for i in range(populationCount):
        newCitizen = np.random.random(2) * lengthOfAxis - edge
        population.append(newCitizen)
    return np.array(population)

generations = np.array([generatePopulation()])

for _ in range(maxGenerations):
    population = np.copy(generations[-1])
    new_population = np.copy(generations[-1])
    for i, x in enumerate(population):
        originalIndices = False
        while not originalIndices:
            r = (*np.trunc(np.random.random(3) * populationCount).astype(int), i)
            originalIndices = len(np.unique(r)) == len(r)
        r1, r2, r3, _ = r
        v = (population[r1] - population[r2]) * mutationConstant + population[r3]
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
    generations = np.concatenate((generations,np.array([new_population])))


X = np.arange(-edge, edge, lengthOfAxis/elementsPerAxis)
Y = np.arange(-edge, edge, lengthOfAxis/elementsPerAxis)
X, Y = np.meshgrid(X,Y)
Z = np.array(myFunctions[selectedFunction](X,Y))

def generateUpdateFromValues(values, ax):
    def update(i):
        print('Showing {0}. generation'.format(i))
        ax.clear()
        ax.plot_surface(X, Y, Z, alpha = 0.6)
        points = values[i]
        points_values = myFunctions[selectedFunction](*np.transpose(points))
        ax.scatter(points[:,0],points[:,1],points_values, color='r')
    return update

def plot3d(values):
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    update = generateUpdateFromValues(values, ax)
    a = anim.FuncAnimation(fig, update, frames = len(generations) - 1, interval = 500, repeat_delay = 2000, repeat = True)
    plt.show()

def generateHeatMapUpdateFromValues(values, ax):
    def update(i):
        print('Showing {0}. generation'.format(i))
        ax.clear()
        im = plt.imshow(Z, cmap='hot', interpolation='nearest', extent=[-edge,edge,-edge,edge])
        points = values[i]
        plt.scatter(points[:,0], points[:,1], color = 'w')
    return update

def plotHeatMap(values):
    fig, ax = plt.subplots()
    im = plt.imshow(Z, cmap='hot', interpolation='nearest', extent=[-edge,edge,-edge,edge])
    update = generateHeatMapUpdateFromValues(values, ax)
    a = anim.FuncAnimation(fig, update, frames = len(values) - 1, interval = 500, repeat_delay = 2000, repeat = True)
    plt.show()

plotHeatMap(generations)
plot3d(generations)
