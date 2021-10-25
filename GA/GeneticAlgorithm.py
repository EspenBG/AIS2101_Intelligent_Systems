import numpy as np
import random


def swap_array(array, p1, p2):
    temp_var = array[p1]
    array[p1] = array[p2]
    array[p2] = temp_var
    return array.copy()


def tournament_selection(num_new: int, population, pop_cost):
    # TODO: This is not tournament_selection yet, only chooses 2 random parents

    pairs_of_parents = num_new//2
    parents = np.zeros((pairs_of_parents, 2, population.shape[1]))

    for i in np.arange(pairs_of_parents):
        for parent in range(2):
            # Select two random parents in the population
            choice1 = random.randint(0, population.shape[0] - 1)
            choice2 = random.randint(0, population.shape[0] - 1)

            rank1 = np.where(pop_cost[:, 0:] == choice1)[0][0]
            rank2 = np.where(pop_cost[:, 0:] == choice2)[0][0]
            if rank1 < rank2:
                parents[i][parent] = population[choice1]
            else:
                parents[i][parent] = population[choice2]




    return parents


# Implement mutation PMX (Partially - Mapped Crossover)
def pmx_crossover(parents, num_of_children):
    # TODO check comments and explanations
    # Iterate thru every set of parents

    children = []
    for (par1, par2) in parents:
        #par1 = np.array((4, 0, 1, 3, 2))
        #par2 = np.array((2, 1, 3, 0, 4))
        # print(par1, par2)
        # Select a random point for crossover
        crossover_point = random.randint(0, len(par1))
        #crossover_point = 3
        # split each parent in two at crossover point

        # Iterate first part from parent 2
        parent1_copy = par1.copy()
        parent2_copy = par2.copy()

        for i in np.arange(crossover_point):
            # for every place check if the index exists in last part of parent 1
            place = par2[i]
            duplicate_at = np.where(parent1_copy == place)
            if duplicate_at[0].shape[0]:
                # exchange the index in the last part to the index in the first part (parent 1)
                parent1_copy = swap_array(parent1_copy, duplicate_at[0], i)
                # print("2")

            place = par1[i]
            duplicate_at = np.where(parent2_copy == place)
            if duplicate_at[0].shape[0]:
                # exchange the index in the last part to the index in the first part (parent 1)
                parent2_copy = swap_array(parent2_copy, duplicate_at[0], i)

        first_child = np.hstack((par2[:crossover_point], parent1_copy[crossover_point:]))
        second_child = np.hstack((par1[:crossover_point], parent2_copy[crossover_point:]))
        children.append(first_child)
        children.append(second_child)
        # print(first_child, second_child, crossover_point)

    # Select the correct number of children, and make it as a numpy array
    children = np.array(children[:][:num_of_children])
    return children


def ox_crossover(parents, num_of_children):
    length = parents.shape[2]

    children = []
    for (par1, par2) in parents:
        # par1 = np.array((1, 4, 5, 9, 7, 2, 0, 6, 8, 3))
        # par2 = np.array((7, 3, 1, 8, 9, 4, 2, 6, 5, 0))
        # length = par1.size

        # Get two crossover points (random place and length)
        place1 = random.randint(0, length - 2)
        place2 = random.randint(place1 + 1, length - 1)

        place1 = 3
        place2 = 6

        # the mid section for each parent is selected

        dupes_par1 = np.in1d(par1, par2[place1:place2])
        dupes_par2 = np.in1d(par2, par1[place1:place2])

        parent1 = par1[dupes_par1.__invert__()].tolist()
        parent2 = par2[dupes_par2.__invert__()].tolist()

        cross_part1 = par1[dupes_par1].tolist()
        cross_part2 = par2[dupes_par2].tolist()
        child1 = []
        child2 = []

        for i in range(length):
            if i < place1:
                child1.append(parent1[0])
                child2.append(parent2[0])
                parent1.pop(0)
                parent2.pop(0)
            elif place1 <= i < place2:
                child1.append(cross_part1[0])
                child2.append(cross_part2[0])
                cross_part1.pop(0)
                cross_part2.pop(0)
            else:
                child1.append(parent1[0])
                child2.append(parent2[0])
                parent1.pop(0)
                parent2.pop(0)

        children.append(child1)
        children.append(child2)

    # Select the correct number of children, and make it as a numpy array
    children = np.array(children[:][:num_of_children])
    # t = np.sort(children, axis=-1)
    return children


def twors_mutation(population, mutation_rate):
    for i in np.arange(population.shape[0]):
        if random.random() <= mutation_rate:
            chromosome = population[i].copy()
            place1 = random.randint(0, chromosome.size-1)
            place2 = random.randint(0, chromosome.size-1)
            chromosome = swap_array(chromosome, place1, place2)
            population[i] = chromosome
    return population


def rsm_mutation(population, mutation_rate):
    for i in np.arange(population.shape[0]):
        if random.random() <= mutation_rate:
            chromosome = population[i].copy()
            place1 = random.randint(0, chromosome.size-1)
            place2 = random.randint(0, chromosome.size-1)
            chromosome = swap_array(chromosome, place1, place2)
            population[i] = chromosome
    return population


class PermutationGA:
    #  Create a GA for permutation problem (tsp)
    def __init__(self, chromosome_len=0, selection_rate=0):
        self.chromosome_len = 0
        self.population = 0
        self.selection_rate = 0
        self.pop_num = 0
        self.elite_num = 0

        self.cost_method = 0
        self.cost_info = np.array(0)
        self.population_cost = 0
        self.mutation_rate = 0.05
        self.crossover = ox_crossover
        self.mutation = twors_mutation
        self.select_new_parents = tournament_selection


    # Generate initial population
    def create_init_pop(self, pop_num):
        available_index = []
        for x in range(0, self.chromosome_len): available_index.append(x)
        population = np.zeros((pop_num, self.chromosome_len), dtype="uint")

        for pop in range(0, pop_num):
            temp_available = available_index[:]

            for place in available_index:
                select = random.choice(temp_available)
                random.randint(0, len(temp_available))
                population[pop, place] = select
                temp_available.pop(temp_available.index(select))

        self.population = population
        self.pop_num = pop_num
        self.population_cost = self.calculate_cost(self.population)


    # Selection rate top 50%? Tournament selection
    def set_selection_sate(self, rate):
        self.selection_rate = rate


    # Elitism
    def set_elitism(self, percent):
        # Set the elitism of the GA, given as a number between 0-1
        # Ex. if the number 0.01 is given the top 1% of the population is always with the system to the next evolution
        # Always keep one if there is set a value not equal 0

        elite_num = int(self.pop_num*percent)
        if elite_num == 0 and percent > 0:
            elite_num = 1

        self.elite_num = elite_num

    def set_chromosome_len(self, new_length):
        self.chromosome_len = new_length

    def set_cost_info(self, permutation_cost, method: str = None):
        self.cost_info = permutation_cost
        if method == "distance":
            self.cost_method = 1

    def evolve(self):
        # Calculate the population cost for the old population
        #self.population_cost = self.calculate_cost(self.population)
        pop_num_no_elite = self.pop_num - self.elite_num
        num_new_chromosome = int(pop_num_no_elite - pop_num_no_elite*self.selection_rate)

        # Get a even number of parents
        num_parents = num_new_chromosome + num_new_chromosome % 2

        # Get the parents to the new population
        new_parents = self.select_new_parents(num_parents, self.population, self.population_cost)

        parents = new_parents.reshape((new_parents.shape[0]*2, self.chromosome_len))
        parents = np.array(parents[:][:num_new_chromosome])

        children = self.crossover(new_parents, num_new_chromosome)
        children = self.mutation(children, self.mutation_rate)
        elites = self.get_elites()

        if (num_new_chromosome*2 + self.elite_num) < self.pop_num:
            num_of_pad = self.pop_num - (num_new_chromosome*2 + self.elite_num)
            padding = self.select_new_parents(num_of_pad + num_of_pad % 2, self.population, self.population_cost)
            padding = padding.reshape((padding.shape[0]*2, self.chromosome_len))
            padding = np.array(padding[:][:num_of_pad])

            new_population = np.vstack((parents, children, elites, padding))

        else:
            new_population = np.vstack((parents, children, elites))


        self.population = new_population.astype(dtype="uint8")
        self.population_cost = self.calculate_cost(self.population)

    def get_elites(self):
        indexes = self.population_cost[:self.elite_num, :1]
        elites = []
        for index in indexes:
            elites.append(self.population[int(index[0])])

        return np.array(elites)

    def get_best(self):
        best_index = self.population_cost[0, 0]
        cost = self.population_cost[0, 1]
        chromosome = self.population[int(best_index)]
        return chromosome, cost

    def get_avg(self):
        return np.average(self.population_cost[:, 1])

    # Implement mutation

    # Cost function
    def cost_function(self, chromosome):
        # only valid solution (travels to all cities and only once)
        # lowest travelled distance (trough all cities and back to the start)
        # use sqrt or only use the square of the distance

        # Check for duplicates
        # Check if every city is visited
        # TODO: add usage of cost method?

        #chromosome = np.array([13, 26, 25, 28, 21, 23, 0, 1, 6, 18, 7, 27, 17, 5, 22, 11, 24, 2, 19, 16, 9, 10, 3, 14, 15, 12, 8, 20, 4])
        place = chromosome
        place_next = np.hstack((chromosome[1:], chromosome[0]))

        dist = (self.cost_info[place_next]-self.cost_info[place])**2

        cost = np.sum(dist)

        # cost = 0
        # for i in np.arange(len(chromosome)):
        #     # Add the euclidean distance
        #     # TODO needs explanation
        #     place = chromosome[i]
        #     next_place = chromosome[(i + 1) % len(chromosome)]
        #     dist = self.cost_info[next_place] - self.cost_info[place]
        #
        #     # cost += np.sqrt(np.sum(dist**2))
        #     cost += np.sum(dist**2)

        return cost

    def calculate_cost(self, population):
        """

        :param population:
        :return: [chromosome, cost, normalized cost]
        """
        # Evaluate the entire population and rank them by cost
        # Set the mean cost
        # Set the best cost


        population_cost = np.zeros((population.shape[0], population.shape[1], 3))
        pop_cost = np.zeros(population.shape[0])

        for i in np.arange(population.shape[0]):
            chromosome = population[i]
            pop_cost[i] = self.cost_function(chromosome)

        try:
            pop_cost_norm = (pop_cost-pop_cost.min())/(pop_cost.max()-pop_cost.min())
        except RuntimeWarning as e:
            print(e)

        population_cost = np.vstack((np.arange(population.shape[0]), pop_cost, pop_cost_norm))
        pop_cost_sorted = np.array(sorted(population_cost.T, key=lambda line: line[2]))
        return pop_cost_sorted


# Normalize to index values...
# De normalize in cost function...
# Set chromosome length
