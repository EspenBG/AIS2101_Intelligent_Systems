import numpy as np
import GeneticAlgorithm as ga
from matplotlib import pylab as plt


def plot_chromosome(chromosome):
    """
    Function to plot the cities and the route of a single chromosome
    :param chromosome: the index values of the cities
    """
    # Enable multithreading for plotting, does not stop the program to plot
    plt.ion()
    fig = plt.figure(1)
    place_cities = fig.add_subplot(111)
    # Plot all the cities
    place_cities.scatter(cities[:, 1], cities[:, 0])
    # place_cities.set_ylim(place_cities.get_ylim()[::-1])
    # Reverse the y- axis to recreate the plot from the ideal solution for western sahara
    place_cities.set_xlim(place_cities.get_xlim()[::-1])
    plt.pause(1)

    for j in np.arange(chromosome.size):
        # Plot the line between each city selected in the chromosome
        line_start = cities[chromosome[j]]
        line_stop = cities[chromosome[(j+1) % chromosome.size]]
        x = [line_start[1], line_stop[1]]
        y = [line_start[0], line_stop[0]]
        place_cities.plot(x, y)

    plt.pause(1)
    plt.show()


# Choose the dataset to be used
city_file = open("wi29.tsp")
# city_file = open("dj38.tsp")

cities = []
# Load the cities from the file
for line in city_file:
    try:
        # This will make errors on every line not containing a city
        city = np.array(line.split(' ')).astype(float)
        # Add the coordinates to the cities array
        cities.append([city[1], city[2]])
    except Exception as e:
        print(e)

# Make the array to an np array
cities = np.array(cities)

# Make the GA and set the parameters and variables
GA = ga.PermutationGA()
GA.set_chromosome_len(len(cities))
GA.set_cost_info(cities, method="distance")
GA.create_init_pop(500)
GA.set_selection_rate(0.5)
GA.set_mutation_rate(0.10)
GA.set_elitism(0.01)

# Set the methods for crossover and mutation
GA.set_crossover_method(ga.ox_crossover)
GA.set_mutation_method(ga.rsm_mutation)

# Arrays to store the best and average cost
best_cost = []
avg_cost = []

for i in range(0, 1000):
    # Evolve the GA to a new generation
    GA.evolve()
    # Get the average cost
    avg_cost.append(GA.get_avg())
    # Get the best cost and chromosome
    best_chromosome, cost = GA.get_best()
    best_cost.append(cost)
    if i % 100 == 0 or i == 999:
        # Plot the best route every 100 generations
        plot_chromosome(best_chromosome)
        print("Iteration: ", i)
        print("Best cost: ", best_cost[i])
        print("Avg cost:  ", avg_cost[i])

# Make a plot for the performance graph of the GA
plt.ioff()  # Comment this out when performance testing, this will block the program from completing.
plt.figure(2)
plt.plot(best_cost)
plt.plot(avg_cost)
plt.legend(("best cost", "average cost"))
plt.xlabel("Iteration")
plt.ylabel("Cost [km]")
plt.show()

# Print the best chromosome
print("Best chromosome: ", best_chromosome)

