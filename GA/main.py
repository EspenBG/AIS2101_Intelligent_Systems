import random
import numpy as np
import sys
import GeneticAlgorithm as GA
from matplotlib import pylab as plt


def plot_chromosome(chromosome):
    plt.ion()
    fig = plt.figure(1)
    place_cities = fig.add_subplot(111)
    place_cities.scatter(cities[:, 1], cities[:, 0])

    # place_cities.set_ylim(place_cities.get_ylim()[::-1])
    place_cities.set_xlim(place_cities.get_xlim()[::-1])
    plt.pause(1)

    for j in np.arange(chromosome.size):
        line_start = cities[chromosome[j]]
        line_stop = cities[chromosome[(j+1) % chromosome.size]]
        x = [line_start[1], line_stop[1]]
        y = [line_start[0], line_stop[0]]

        place_cities.plot(x, y)
    plt.pause(1)
    plt.show()

city_file = open("wi29.tsp")
# cities = np.loadtxt("wi29.tsp", delimiter=" ")
cities = []
for line in city_file:
    try:
        city = np.array(line.split(' ')).astype(float)
        cities.append([city[1], city[2]])
    except Exception as e:
        print(e)

print(cities)
cities = np.array(cities)

GA = GA.PermutationGA()
GA.set_chromosome_len(len(cities))
GA.set_cost_info(cities, method="distance")
#GA.set_chromosome_len(5)
GA.create_init_pop(500)
GA.set_selection_sate(0.5)
GA.set_elitism(0.01)


for i in range(0, 1000):
    GA.evolve()
    if i % 100 == 0:
        best_chromosome, cost = GA.get_best()
        plot_chromosome(best_chromosome)
        print("Best cost: ", cost)
        print("Avg cost:  ", GA.get_avg())


print(cost)

# print(cities)
# Find the chromosome length

# Set the cost function


# plot the locations of the cities
# plot the best solution each generation

