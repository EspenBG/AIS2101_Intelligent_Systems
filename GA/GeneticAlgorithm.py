import numpy as np
import random


def swap_array(array, p1, p2):
    """
    Function to swap to indexes in a Numpy array
    :param array: Numpy array
    :param p1: index number for place 1
    :param p2: index number for place 2
    :return: Copy of the new array
    """
    temp_var = array[p1]
    array[p1] = array[p2]
    array[p2] = temp_var
    return array.copy()


def swap_section(array, place1, place2):
    """
    Function to reverse a section of an Numpy array
    :param array: Numpy array
    :param place1: index number for section start
    :param place2: index number for section end
    :return: Copy of the new array
    """
    # Get the subsection to be reversed
    subsection = array[place1:place2]
    # reverse the subsection
    subsection = subsection[::-1]
    # Make the new array with the reversed section
    array = np.hstack((array[0:place1], subsection, array[place2:]))
    return array.copy()


def tournament_selection(num_new: int, population, pop_cost):
    """
    Function to select parents from a population using Tournament selection.
    :param num_new: The number of new children that needs to be made.
    :param population: The entire population in an Numpy array
    :param pop_cost: The cost of the population in a numpy array.
    :return: A numpy array containing pairs of parents
    """
    # Get how many pairs of parents that there should be
    pairs_of_parents = num_new//2
    # make a placeholder array for the parents
    parents = np.zeros((pairs_of_parents, 2, population.shape[1]))

    for i in np.arange(pairs_of_parents):
        # Select one parent at a time
        for parent in range(2):
            # Select two random parents in the population
            choice1 = random.randint(0, population.shape[0] - 1)
            choice2 = random.randint(0, population.shape[0] - 1)
            # Get the cost of both the choices
            rank1 = np.where(pop_cost[:, 0:] == choice1)[0][0]
            rank2 = np.where(pop_cost[:, 0:] == choice2)[0][0]
            # Select the choice with the highest cost
            if rank1 < rank2:
                parents[i][parent] = population[choice1]
            else:
                parents[i][parent] = population[choice2]

    # Return the pairs of parents
    return parents


# Implement crossover PMX (Partially - Mapped Crossover)
def pmx_crossover(parents, num_of_children):
    """
    Function to implement PMX crossover.
    :param parents: Array with pairs of parents
    :param num_of_children: number of children
    :return: children
    """

    children = []
    # Iterate thru every set of parents
    for (par1, par2) in parents:
        # par1 = np.array((4, 0, 1, 3, 2))  # Used for testing
        # par2 = np.array((2, 1, 3, 0, 4))  # Used for testing

        # Select a random point for crossover
        crossover_point = random.randint(0, len(par1) - 1)

        # crossover_point = 3 # Used for testing

        # Make a copy of each parent
        parent1_copy = par1.copy()
        parent2_copy = par2.copy()

        # iterate thru every index in the crossover section
        for i in np.arange(crossover_point):
            # for every place check if the index exists in parent 1
            place = par2[i]
            # Find duplicates
            duplicate_at = np.where(parent1_copy == place)
            if duplicate_at[0].shape[0]:
                # exchange the index in the last part to the index in the first part (parent 1)
                parent1_copy = swap_array(parent1_copy, duplicate_at[0], i)

            # for every place check if the index exists in parent 2
            place = par1[i]
            # Find duplicates
            duplicate_at = np.where(parent2_copy == place)
            if duplicate_at[0].shape[0]:
                # exchange the index in the last part to the index in the first part (parent 2)
                parent2_copy = swap_array(parent2_copy, duplicate_at[0], i)

        # Make the children and add them to the array
        first_child = np.hstack((par2[:crossover_point], parent1_copy[crossover_point:]))
        second_child = np.hstack((par1[:crossover_point], parent2_copy[crossover_point:]))
        children.append(first_child)
        children.append(second_child)

    # Select the correct number of children, and make it as a numpy array
    children = np.array(children[:][:num_of_children])
    return children


def ox_crossover(parents, num_of_children):
    """
    Function to implement OX crossover.
    :param parents: Array with pairs of parents
    :param num_of_children: number of children
    :return: children
    """
    # Find the length of each chromosome
    length = parents.shape[2]
    children = []

    # Iterate thru every pair of parents
    for (par1, par2) in parents:
        # par1 = np.array((1, 4, 5, 9, 7, 2, 0, 6, 8, 3)) # Used for testing
        # par2 = np.array((7, 3, 1, 8, 9, 4, 2, 6, 5, 0)) # Used for testing
        # length = par1.size # Used for testing

        # Get two crossover points (random place and length)
        place1 = random.randint(0, length - 2)
        place2 = random.randint(place1 + 1, length - 1)

        # Find all the duplicates in the crossover section and the other parent
        dupes_par1 = np.in1d(par1, par2[place1:place2])
        dupes_par2 = np.in1d(par2, par1[place1:place2])

        # Get the parent without the duplicates
        parent1 = par1[dupes_par1.__invert__()].tolist()
        parent2 = par2[dupes_par2.__invert__()].tolist()
        # Get the crossover section
        cross_part1 = par1[dupes_par1].tolist()
        cross_part2 = par2[dupes_par2].tolist()
        child1 = []
        child2 = []

        # Make the new children
        for i in range(length):
            if i < place1:
                # Before the crossover section
                # Select values from parent without dupes
                child1.append(parent1[0])
                child2.append(parent2[0])
                # Remove the values added
                parent1.pop(0)
                parent2.pop(0)
            elif place1 <= i < place2:
                # The crossover section
                # Select values from crossover section
                child1.append(cross_part1[0])
                child2.append(cross_part2[0])
                # Remove the values added
                cross_part1.pop(0)
                cross_part2.pop(0)
            else:
                # After the crossover section
                # Select values from parent without dupes
                child1.append(parent1[0])
                child2.append(parent2[0])
                # Remove the values added
                parent1.pop(0)
                parent2.pop(0)

        # Add the children to the array
        children.append(child1)
        children.append(child2)

    # Select the correct number of children, and make it as a numpy array
    children = np.array(children[:][:num_of_children])
    # t = np.sort(children, axis=-1)
    return children


def twors_mutation(population, mutation_rate):
    """
    Function to implement Twors mutation on a population.
    The Twors mutation is a simple mutation method where if a chromosome should bemutated, it selects
    two random  indexes in the chromosome and swaps them. This mutation method only allows for one change
    per chromosome, and this can limit the exploration characteristic for the algorithm.
    :param population: Array containing the population
    :param mutation_rate: The mutation rate as a number between 0 and 1
    :return: mutated population
    """
    # Iterate thru the entire population
    for i in np.arange(population.shape[0]):
        # Randomly choose from the population with the probability of the mutation rate
        if random.random() <= mutation_rate:
            # Make a copy of the chromosome
            chromosome = population[i].copy()
            # Select the tw points that should be swapped
            place1 = random.randint(0, chromosome.size-1)
            place2 = random.randint(0, chromosome.size-1)
            # Swap the indexes and add the mutated chromosome back
            chromosome = swap_array(chromosome, place1, place2)
            population[i] = chromosome
    # Return the mutated population
    return population


def psm_mutation(population, mutation_rate):
    """
    Function to implement partial shuffle mutation.
    The next method is the partial shuffle mutation and it is similar to Twos, but instead of only
    swapping one pair of indexes it can swap multiple pairs.
    :param population: Array containing the population
    :param mutation_rate: The mutation rate as a number between 0 and 1
    :return: mutated population
    """
    # Iterate thru the entire population
    for i in np.arange(population.shape[0]):
        # Randomly choose from the population with the probability of the mutation rate
        if random.random() <= mutation_rate:
            chromosome = population[i].copy()

            # Randomly generate a value between 0 and 1 for every index
            mutate_chance = np.random.random(chromosome.size)

            for j in np.arange(chromosome.size):
                # Mutate all the indexes with mutate chance lower than the mutation rate
                if mutate_chance[j] <= mutation_rate:
                    place1 = j
                    # Find a random index to swap with
                    place2 = random.randint(0, chromosome.size-1)
                    # Swap the places
                    chromosome = swap_array(chromosome, place1, place2)
            # Add the mutated chromosome back in the population
            population[i] = chromosome
    # Return the mutated population
    return population


def rsm_mutation(population, mutation_rate):
    """
    Function to implement Reverse sequence mutation.
    This is method were each chromosome is mutated with the probability of the mutation rate.
    If a chromosome is chosen to be mutated, two random indexes are chosen.
    The section that is between these two point is reversed.
    :param population: Array containing the population
    :param mutation_rate: The mutation rate as a number between 0 and 1
    :return: mutated population
    """
    # loop thru every element in in the population
    for i in np.arange(population.shape[0]):
        # Randomly check if the mutation should happen
        if random.random() <= mutation_rate:
            # Make a copy of the chromosome
            chromosome = population[i].copy()
            # Select two random places in the chromosome
            place1 = random.randint(0, chromosome.size-2)
            place2 = random.randint(place1, chromosome.size-1)
            # Swap the section between the two places
            chromosome = swap_section(chromosome, place1, place2)
            population[i] = chromosome
    # Return the mutated population
    return population


class PermutationGA:
    """
    CLASS FOR SOLVING A PERMUTATION PROBLEM WITH A GENETIC ALGORITHM.
    """
    #  Create a GA for permutation problem (tsp)
    def __init__(self, chromosome_len=0, selection_rate=0):
        """
        Initialize an instance of the class, with the parameters chromosome_len and selection_rate.
        :param chromosome_len: The length of each chromosome (optional)
        :param selection_rate: The selection rate of the GA (optional)
        """
        # Initialize the variables for the object
        self.chromosome_len = chromosome_len
        self.population = 0
        self.selection_rate = selection_rate
        self.pop_num = 0
        self.elite_num = 0
        self.cost_method = 0
        self.cost_info = np.array(0)
        self.population_cost = 0
        self.mutation_rate = 0.05
        # Set the default functions
        self.crossover = ox_crossover
        self.mutation = psm_mutation
        self.select_new_parents = tournament_selection

    # Selection rate top 50%? Tournament selection
    """SETTERS"""
    def set_selection_rate(self, rate):
        """
        Function to set a new selection rate for the GA
        :param rate: New selection rate (number between 0 and 1)
        """
        self.selection_rate = rate

    def set_mutation_rate(self, rate):
        """
        Function to set a new mutation rate for the GA
        :param rate: new mutation rate (number between 0 and 1)
        """
        self.mutation_rate = rate

    def set_mutation_method(self, method):
        """
        Function to set the mutation method.
        Pre-made functions that can be used is: twors_mutation, psm_mutation and rsm_mutation
        :param method: This need to be a function
        """
        self.mutation = method

    def set_crossover_method(self, method):
        """
        Function to set the crossover method.
        Pre-made functions that can be used is: pmx_crossover and ox_crossover
        :param method: This need to be a function
        """
        self.crossover = method

    def set_selection_method(self, method):
        """
        Function to set the selection method used to select new parents
        Pre-made functions that can be used is: tournament_selection
        :param method: This need to be a function
        """
        self.select_new_parents = method

    def set_elitism(self, percent):
        """
        Function to set the elitism of the GA.
        Set the elitism of the GA, given as a number between 0-1
        Ex. if the number 0.01 is given the top 1% of the population is always with the system to the next evolution
        Always keep one if there is set a value not equal 0
        :param percent: number between 0-1
        """
        # Calculate the number of elites that is supposed to be used
        elite_num = int(self.pop_num*percent)
        # Make sure that there is always one elite, unless the percentage is zero
        if elite_num == 0 and percent > 0:
            elite_num = 1

        self.elite_num = elite_num

    def set_chromosome_len(self, new_length):
        """
        Function to set a new chromosome length.
        :param new_length: Integer over 0
        """
        self.chromosome_len = new_length

    def set_cost_info(self, permutation_cost, method: str = None):
        """
        Function to set the cost information and method to the GA.
        The permutation_cost should be a Numpy array with the x and y positions to all the cities
        :param permutation_cost: Numpy array with the placement of the cities
        :param method: set to "distance" to use the euclidean distance
        """
        self.cost_info = permutation_cost
        # set the cost method if selected
        if method == "distance":
            self.cost_method = 1

    """GETTERS"""
    def get_elites(self):
        """
        Function to get the elites in the population (the ones with the lowest cost)
        :return: Numpy array of elites
        """
        # Find the indexes in the population that have the lowest cost
        indexes = self.population_cost[:self.elite_num, :1]
        elites = []

        # Add all the elites to the array
        for index in indexes:
            elites.append(self.population[int(index[0])])

        # Return the array
        return np.array(elites)

    def get_best(self):
        """
        Function to get the best chromosome, and the cost
        :return: (chromosome, cost) tuple
        """
        # The index for the best chromosome is stored in (0, 0)
        best_index = self.population_cost[0, 0]
        # Get the cost
        cost = self.population_cost[0, 1]
        # Get the chromosome from the index
        chromosome = self.population[int(best_index)]
        # Return the chromosome and cost
        return chromosome, cost

    def get_avg(self):
        """
        Function to get the average cost of the entire population
        :return:
        """
        # Calculate and return the average cost
        return np.average(self.population_cost[:, 1])

    """FUNCTIONS"""
    def create_init_pop(self, pop_num):
        """
        Function to generate a random initial population.
        :param pop_num: The number of chromosomes in the population
        """
        # Make an array contain all the indexes for one permutation
        available_index = []
        for x in range(0, self.chromosome_len): available_index.append(x)

        # Make a placeholder array for the entire population
        population = np.zeros((pop_num, self.chromosome_len), dtype="uint")

        for pop in range(0, pop_num):
            # Make a copy of the indexes
            temp_available = available_index[:]

            for place in available_index:
                # Select a random value in the available indexes
                select = random.choice(temp_available)
                # Add the value in the population
                population[pop, place] = select
                # Remove the value from the temp
                temp_available.pop(temp_available.index(select))

        # Set the new population in the object
        self.population = population
        self.pop_num = pop_num
        # Calculate the cost of the initial population
        self.population_cost = self.calculate_cost(self.population)

    def evolve(self):
        """
        Function to evolve the population to a new generation.
        """
        # Find the population number without the elites
        pop_num_no_elite = self.pop_num - self.elite_num
        # Find the number of children  that is needed
        num_new_chromosome = int(pop_num_no_elite - pop_num_no_elite*self.selection_rate)

        # Get a even number of parents
        num_parents = num_new_chromosome + num_new_chromosome % 2

        # Get the parents to the new population
        new_parents = self.select_new_parents(num_parents, self.population, self.population_cost)

        # Get the new children
        children = self.crossover(new_parents, num_new_chromosome)
        # Mutate the new chromosome
        children = self.mutation(children, self.mutation_rate)

        # Reformat the parents to a uniform shape
        parents = new_parents.reshape((new_parents.shape[0]*2, self.chromosome_len))
        # Limit the number of parents to the same number as children (will at most remove 1)
        parents = np.array(parents[:][:num_new_chromosome])

        # Get the elites in the population
        elites = self.get_elites()

        # Make the new population based on how many chromosomes need to be padded
        if (num_new_chromosome*2 + self.elite_num) < self.pop_num:
            # There are not enough chromosomes
            num_of_pad = self.pop_num - (num_new_chromosome*2 + self.elite_num)
            # Use the selection method to get more chromosomes from the old population
            padding = self.select_new_parents(num_of_pad + num_of_pad % 2, self.population, self.population_cost)
            padding = padding.reshape((padding.shape[0]*2, self.chromosome_len))
            padding = np.array(padding[:][:num_of_pad])

            # Add all the chromosomes to a new population
            new_population = np.vstack((parents, children, elites, padding))

        elif (num_new_chromosome*2 + self.elite_num) > self.pop_num:
            # There are to many chromosomes
            # Constrain the amount of parents
            parents = np.array(parents[:][:self.pop_num - num_new_chromosome - self.elite_num])

            # Add all the chromosomes to a new population
            new_population = np.vstack((parents, children, elites))
        else:
            # Add all the chromosomes to a new population
            new_population = np.vstack((parents, children, elites))

        # Set the new generation as the current population and calculate the cost
        self.population = new_population.astype(dtype="uint8")
        self.population_cost = self.calculate_cost(self.population)

    def cost_function(self, chromosome):
        """
        The cost function for the GA. Returns the cost.
        Calculates the lowest travelled distance (trough all cities and back to the start) given a chromosome.
        :param chromosome: The indexes for the cities
        :return: cost
        """
        # Get all the places in the chromosome (i)
        place = chromosome
        # Get the placement of the next city (i+1)
        place_next = np.hstack((chromosome[1:], chromosome[0]))

        # Find the squared distance between all the points
        dist_squared = np.sum((self.cost_info[place_next] - self.cost_info[place]) ** 2, axis=1)
        # Check if the euclidean distance should be used or only the square
        if self.cost_method == 0:
            # Only use the square
            cost = np.sum(dist_squared)
        else:
            # Use the euclidean distance
            dist = dist_squared**(1/2)
            cost = np.sum(dist)
        return cost

    def calculate_cost(self, population):
        """
        Function to calculate the cost of a population of chromosomes.
        Evaluate the entire population and ranks them by cost.
        :param population:
        :return: Numpy array [chromosome, cost, normalized cost]
        """
        # Create an array to store all the cost information
        pop_cost = np.zeros(population.shape[0])

        # Iterate thru the entire population
        for i in np.arange(population.shape[0]):
            # Choose the chromosome
            chromosome = population[i]
            # Get the cost for the chromosome
            pop_cost[i] = self.cost_function(chromosome)

        try:
            # Try to normalize the cost
            pop_cost_norm = (pop_cost-pop_cost.min())/(pop_cost.max()-pop_cost.min())
        except RuntimeWarning as e:
            print(e)

        # Create the complete population cost array containing the index in pop, pop cost, and norm cost
        population_cost = np.vstack((np.arange(population.shape[0]), pop_cost, pop_cost_norm))
        # sort the entire cost array by the normalized cost
        pop_cost_sorted = np.array(sorted(population_cost.T, key=lambda line: line[2]))
        return pop_cost_sorted
