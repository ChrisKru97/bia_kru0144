import matplotlib.pyplot as plt 
import matplotlib.animation as anim
import numpy as np

selectedFunction="levy"
iterations=20
pointsPerIteration=100
lengthOfAxis=40
startingPoint=(-15,-15)
elementsPerAxis=1000

#hill climb, simulated annealing
sigma=lengthOfAxis/20

#simulated annealing
t_0 = 120
t_min = 0.1
alpha = 0.96

#function variables
d = 2 
m = 10
a = 20
b = 0.2
c = 2*np.pi

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

edge = lengthOfAxis / 2

# Checking for going out of boundaries
# Omitting that may disrupt fluidness of the animation 
isOutOfBounds = lambda x, y: x < -edge or x > edge or y > edge or y < -edge 

X = np.arange(-edge, edge, lengthOfAxis/elementsPerAxis)
Y = np.arange(-edge, edge, lengthOfAxis/elementsPerAxis)
X, Y = np.meshgrid(X,Y)
Z = np.array(myFunctions[selectedFunction](X,Y))

def generateUpdateFromValues(values, ax):
    def update(i):
        ax.clear()
        ax.plot_surface(X, Y, Z, alpha=0.6)
        currentValues = values[i]
        ax.scatter([currentValues[0]],[currentValues[1]],[currentValues[2]], color='r')
    return update

def plot3d(values):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    update = generateUpdateFromValues(values, ax)
    a = anim.FuncAnimation(fig, update, frames = len(values) - 1, repeat=True)
    plt.show()
    
def generateHeatMapUpdateFromValues(values, im, pointValue):
    pointDiameter = np.around(elementsPerAxis/100).astype(int)
    def update(i):
        currentData = np.copy(Z)
        currentX = np.around(((values[i][0]/lengthOfAxis)+0.5)*elementsPerAxis).astype(int) - pointDiameter
        currentY = np.around(((values[i][1]/lengthOfAxis)+0.5)*elementsPerAxis).astype(int) - pointDiameter
        currentData[currentX:currentX+pointDiameter*2,currentY:currentY+pointDiameter*2] = -1000
        im.set_array(currentData)
    return update

def plotHeatMap(values):
    Zmax = np.max(Z)
    fig = plt.figure()
    im = plt.imshow(Z, cmap='hot', interpolation='nearest', extent=[-edge,edge,-edge,edge]) 
    update = generateHeatMapUpdateFromValues(values, im, Zmax)
    a = anim.FuncAnimation(fig, update, frames = len(values) - 1, repeat=True)
    plt.show()

def generateBlindSearchSteps():
    guesses = np.random.random((iterations*pointsPerIteration,X.ndim))*20 -10
    minValue = np.Inf
    values = np.array([])
    for x,y in guesses:
        value = np.array(myFunctions[selectedFunction](x,y))
        if value < minValue:
            values = np.concatenate((values,np.array([value])))
            minValue = value
    return values

def generateHillClimbSteps():
    lastMinPoint = np.array([startingPoint[0],startingPoint[1],myFunctions[selectedFunction](startingPoint[0],startingPoint[1])])
    minPoints = np.array([lastMinPoint]) 
    for i in range(iterations):
        guesses = np.random.normal((lastMinPoint[0],lastMinPoint[1]),sigma,(pointsPerIteration,X.ndim))
        minPoint = lastMinPoint
        for x,y in guesses:
            if not isOutOfBounds(x,y):
                value = myFunctions[selectedFunction](x,y)
                if value < minPoint[2]:
                    minPoint = np.array([x,y,value])
        if minPoint[2] < lastMinPoint[2]:
            minPoints = np.concatenate((minPoints, np.array([minPoint])))
            lastMinPoint = minPoint
    return minPoints

def generateSimulatedAnnealingSteps():
    lastMinPoint = np.array([*startingPoint[:2],myFunctions[selectedFunction](*startingPoint[:2])])
    values = np.array([lastMinPoint])
    t = t_0
    shouldAdd = False
    while t > t_min:
        guess = np.random.normal(lastMinPoint[:2],sigma,X.ndim)
        if isOutOfBounds(*guess):
            continue
        value = myFunctions[selectedFunction](*guess)
        if value < lastMinPoint[2]:
            shouldAdd = True
        else:
            r = np.random.random()
            if r < np.exp(-((value-lastMinPoint[2])/t)):
                shouldAdd = True
        if shouldAdd:
            minPoint = np.array([*guess,value])
            values = np.concatenate((values, np.array([minPoint])))
            lastMinPoint = minPoint
            shouldAdd = False
        t = t * alpha 
    return values

values = generateSimulatedAnnealingSteps()

if(len(values) > 1):
    plot3d(values)
    plotHeatMap(values)
else:
    print("No values, probably went out of boundaries")
