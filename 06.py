import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np

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

get_w = lambda i: w_s * (w_s - w_e) * i / M_max

# Function variables
d = 2
m = 10
a = 20
b = 0.2
c = 2 * np.pi

def putToBounds(individual):
    individual = np.where(individual < -edge, -edge, individual)
    individual = np.where(individual > edge, edge, individual)
    return individual

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

def generateSwarm():
    swarm = []
    for _ in range(pop_size):
        individual = np.random.rand(d) * lengthOfAxis - edge
        velocity = np.array([0,0]) # could be random, but inertia_weight is 0 for first iteration which will cancel the initial velocity
        value = countObjectiveValue(individual)

        # every individual in form:
        #  [0] => actual vector with its function value in last position
        #  [1] => actual velocity
        #  [2] => individual best position
        swarm.append(np.array([np.array([*individual, value]), velocity, np.array([*individual, value])], dtype=object))
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
        next_v = inertia_weight * x[1] + c1 * np.random.uniform() * (x[2][:2] - x[0][:2]) + c2 * np.random.uniform() * (gBest[:2] - x[0][:2])
        next_v = np.where(next_v < v_min, v_min, next_v)
        next_v = np.where(next_v > v_max, v_max, next_v)
        new_pos = putToBounds(x[0][:2] + next_v)
        new_pos_value = countObjectiveValue(new_pos)
        if(new_pos_value < x[2][2]):
            next_x = np.array([np.array([*new_pos, new_pos_value]), next_v, np.array([*new_pos, new_pos_value])], dtype=object)
        else:
            next_x = np.array([np.array([*new_pos, new_pos_value]), next_v, x[2]], dtype=object)    
        if(new_pos_value < gBest[2]):
            gBest = np.array([*new_pos, new_pos_value])
        swarm[i] = next_x
    migrations = np.concatenate((migrations, np.array([swarm], copy = True)))


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
        ax.scatter(points[:,0], points[:,1], points[:,2], color='r')
    return update

def plot3d(values):
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    update = generateUpdateFromValues(values, ax)
    a = anim.FuncAnimation(fig, update, frames = len(migrations) - 1, interval = 500, repeat_delay = 2000, repeat = True)
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

# forgetting the velocities and p_best
migrations = np.array(list(map(lambda x1: np.array(list(map(lambda x2: x2[0], x1))), migrations)))

plotHeatMap(migrations)
plot3d(migrations)