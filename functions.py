import numpy as np

d = 2
m = 10
a = 20
b = 0.2
c = 2 * np.pi


def generateIndexes(shape):
    d = shape[0]
    rest = shape[1:]
    indexes = []
    for i in range(d):
        indexes.append(np.full(rest, i + 1))
    return np.array(indexes)


def sphere(*x):
    return (np.array(x)**2).sum(axis=0)


def schwefel(*x):
    vec = np.array(x)
    d = vec.shape[0]
    return 418.9829 * d - (vec * np.sin(np.sqrt(np.abs(vec)))).sum(axis=0)


def rosenbrock(*x):
    vec = np.array(x)
    first = vec[:-1]
    second = vec[1:]
    return (100 * (second - first ** 2) ** 2 + (first - 1) ** 2).sum(axis=0)


def rastrigin(*x):
    vec = np.array(x)
    d = vec.shape[0]
    return 10 * d + (vec ** 2 - 10 * np.cos(2 * np.pi * vec)).sum(axis=0)


def griewank(*x):
    vec = np.array(x)
    indexes = generateIndexes(vec.shape)
    return (vec ** 2 / 4000).sum(axis=0) - (np.cos(vec / np.sqrt(indexes))).prod(axis=0) + 1


def levy(*x):
    vec = np.array(x)
    w = 1 + (vec[:-1] - 1) / 4
    w_d = 1 + (vec[-1] - 1) / 4
    return np.sin(np.pi * w[0]) ** 2 + ((w - 1) ** 2 * (1 + 10 * np.sin(np.pi * w + 1) ** 2)).sum(axis=0) + (w_d - 1) ** 2 * (1 + np.sin(2 * np.pi * w_d + 1))


def michalewicz(*x):
    vec = np.array(x)
    indexes = generateIndexes(vec.shape)
    return - (np.sin(vec) * np.sin(indexes * vec ** 2 / np.pi) ** (2 * m)).sum(axis=0)


def zakharov(*x):
    vec = np.array(x)
    indexes = generateIndexes(vec.shape)
    return (vec ** 2).sum(axis=0) + (0.5 * indexes * vec).sum(axis=0) ** 2 + (0.5 * indexes * vec).sum(axis=0) ** 4


def ackley(*x):
    vec = np.array(x)
    d = vec.shape[0]
    return -a * np.exp(-b * np.sqrt((vec ** 2).sum(axis=0) / d)) - np.exp((np.cos(c * vec)).sum(axis=0) / d) + a + np.e


functions = {
    "sphere": sphere,
    "schwefel": schwefel,
    "rosenbrock": rosenbrock,
    "rastrigin": rastrigin,
    "griewank": griewank,
    "levy": levy,
    "michalewicz": michalewicz,
    "zakharov": zakharov,
    "ackley": ackley,
}
