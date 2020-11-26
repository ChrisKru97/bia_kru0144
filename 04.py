import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np
import requests
import re

populationCount = 50
citiesCount = 15
generationCount = 200
axisLength = 100


def generateNames(n):
    # return [chr(x) for x in range(97, 123)][:n]
    r = requests.get('https://en.wikipedia.org/wiki/List_of_largest_cities')
    m = np.array(re.findall(
        r'<tr>\n<td align="left"><a href="\/wiki\/.+" title=".+">(.+)<\/a>', r.text))
    return m[:n]


def countTotalDistance(citizen, cities):
    distance = 0
    for i, city in enumerate(citizen):
        if i < citiesCount:
            distance += np.sqrt(np.sum((cities[city] -
                                        cities[citizen[i+1]])**2))
    return distance


def generatePopulation(citiesList):
    population = []
    startingCity = citiesList[0]
    for i in range(populationCount):
        newCitizen = np.copy(citiesList[1:])
        np.random.shuffle(newCitizen)
        newCitizen = np.concatenate(
            ([startingCity], newCitizen, [startingCity]))
        population.append(newCitizen)
    return np.array(population)


def generateCities(names):
    randomPositions = np.random.random((citiesCount, 2))*200
    citiesPositions = {}
    for i, city in enumerate(names):
        citiesPositions[city] = randomPositions[i]
    return citiesPositions


def crossover(parent_A, parent_B):
    slicePoint = np.trunc(np.random.random()*(citiesCount-3)+2).astype(int)
    offspring = np.copy(parent_A[:slicePoint])
    offspring_rest = parent_A[slicePoint:-1]
    B_part = np.copy(parent_B[slicePoint:-1])
    available = []
    for city in offspring_rest:
        if city not in B_part:
            available.append(city)
    if len(available) > 0:
        for i, city in enumerate(B_part):
            if city in offspring:
                B_part[i] = available.pop()
    return np.concatenate((offspring, B_part, [offspring[0]]))


def mutate(offspring):
    a = np.trunc((np.random.random()*citiesCount-1) + 1).astype(int)
    b = a
    while b == a:
        b = np.trunc((np.random.random()*citiesCount-1) + 1).astype(int)
    offspring[b], offspring[a] = offspring[a], offspring[b]
    return offspring


def findBest(population, index):
    bestLength = np.Inf
    for citizen in population:
        length = countTotalDistance(citizen, cities)
        if length < bestLength:
            bestCitizen = citizen
            bestLength = length
    print('Best of generation {0} with length: {1}'.format(index, bestLength))
    return bestCitizen, bestLength


names = generateNames(citiesCount)
cities = generateCities(names)
population = generatePopulation(names)
new_population = np.copy(population)

bestCitizen, newBestLength = findBest(population, 'initial')
bestOfGeneration = np.array([bestCitizen])

for i in range(generationCount):
    for j in range(populationCount):
        parent_A = population[j]
        B_index = j
        while B_index == j:
            B_index = np.trunc(np.random.random()*20).astype(int)
        parent_B = population[B_index]

        offspring_AB = crossover(parent_A, parent_B)
        offspring_AB = mutate(offspring_AB)

        offspring_distance = countTotalDistance(offspring_AB, cities)
        parent_distance = countTotalDistance(parent_A, cities)

        if offspring_distance <= parent_distance:
            new_population[j] = offspring_AB
    population = np.copy(new_population)
    bestCitizen, bestLength = findBest(population, i)
    if bestLength < newBestLength:
        bestOfGeneration = np.concatenate(
            (bestOfGeneration, np.array([bestCitizen])))
        newBestLength = bestLength

fig, ax = plt.subplots()


def update(i):
    print('Showing {0}. of generations with good mutation'.format(i))
    ax.clear()
    current = bestOfGeneration[i]
    toPlot = np.array(list(map(lambda x: cities[x], current)))
    ax.plot(*np.transpose(toPlot), '-or')
    for i, txt in enumerate(current):
        ax.annotate(txt, toPlot[i])


a = anim.FuncAnimation(fig, update, interval=500, repeat_delay=2000, frames=len(
    bestOfGeneration)-1, repeat=True)
plt.show()
