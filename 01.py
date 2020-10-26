import matplotlib.pyplot as plt 
import matplotlib.animation as anim
import numpy as np

selectedFunction=3
maxGuesses=10000

d = 2
m = 10
a = 20
b = 0.2
c = 2*np.pi

class Function:
    def __init__(self, name):
        self.name = name

    def sphere(params):
        sum=0
        for p in params:
            sum+= p*p
        return sum

def levy(x,y):
    w = [1+(x-1)/4,1+(y-1)/4]
    return np.sin(np.pi*w[0])**2 + (w[0]-1)**2 * (1+10*np.sin(np.pi*w[0]+1)**2) + (w[1]-1)**2 * (1+10*np.sin(np.pi*w[1]+1)**2) + (w[1]-1)**2 * (1+np.sin(2*np.pi*w[1])**2)

myFunc = [lambda x, y : x*x + y*y, #sphere
        lambda x, y: 418.9829*d - (x*np.sin(np.sqrt(np.absolute(x)))+y*np.sin(np.sqrt(np.absolute(y)))), # schwefel
        lambda x,y: (100*(y-x*x)**2+(x-1)**2), #rosenbrock
        lambda x,y: 10*d + (x*x - 10*np.cos(2*np.pi*x))+(y*y - 10*np.cos(2*np.pi*y)), #rastrigin
        lambda x,y: ((x*x)/4000+(y*y)/4000)-((np.cos(x))*(np.cos(y/np.sqrt(2)))+1), #griewank
        levy, #levy
        lambda x,y: -(np.sin(x)*np.sin((x*x)/np.pi)**(2*m))-(np.sin(y)*np.sin((2*y*y)/(np.pi))**(2*m)), #michalewicz
        lambda x,y: x*x + y*y + (0.5*x + y)**2 + (0.5*x+y)**4, #zakharov
        lambda x,y: -a*np.exp(-b*np.sqrt((x*x+y*y)/2))-np.exp((np.cos(c*x)+np.cos(c*y))/2)+a+np.exp(1), #ackley
        ]

X = np.arange(-10, 10, 0.5)
Y = np.arange(-10, 10, 0.5)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

X, Y = np.meshgrid(X,Y)

Z = myFunc[selectedFunction](X,Y) 

guesses = np.random.random((maxGuesses,X.ndim))*20 - 10

minValue = np.Inf 

values = []

for x,y in guesses:
    value = myFunc[selectedFunction](x,y)
    if value < minValue:
        values.append([x,y,value])
        minValue = value

def update(i):
    ax.clear()
    ax.plot_surface(X, Y, Z, alpha=0.6)
    currentValues = values[i]
    ax.plot([currentValues[0]],[currentValues[1]],[currentValues[2]],markerfacecolor='r', markeredgecolor='r', marker='o', markersize=4 )

a = anim.FuncAnimation(fig, update, frames = len(values) -1 , repeat=True)
plt.show()
