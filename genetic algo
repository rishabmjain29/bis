import random

# Define the fitness function
def fitness_function(x):
    return x**2

# Generate initial population
def generate_population(size, lower_bound, upper_bound):
    return [random.randint(lower_bound, upper_bound) for _ in range(size)]

# Select parents based on fitness
def select_parents(population):
    return random.choices(population, weights=[fitness_function(x) for x in population], k=2)

# Perform crossover between two parents
def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(bin(parent1)) - 2)
    mask = (1 << crossover_point) - 1
    child1 = (parent1 & mask) | (parent2 & ~mask)
    child2 = (parent2 & mask) | (parent1 & ~mask)
    return child1, child2

# Apply mutation to introduce variation
def mutate(individual, mutation_rate, lower_bound, upper_bound):
    if random.random() < mutation_rate:
        individual = random.randint(lower_bound, upper_bound)
    return individual

# Genetic Algorithm
def genetic_algorithm(pop_size, generations, lower_bound, upper_bound, mutation_rate):
    population = generate_population(pop_size, lower_bound, upper_bound)
    for generation in range(generations):
        new_population = []
        for _ in range(pop_size // 2):
            parent1, parent2 = select_parents(population)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1, mutation_rate, lower_bound, upper_bound)
            child2 = mutate(child2, mutation_rate, lower_bound, upper_bound)
            new_population.extend([child1, child2])
        population = new_population
        best_individual = max(population, key=fitness_function)
        print(f"Generation {generation + 1}: Best Fitness = {fitness_function(best_individual)}")
    return max(population, key=fitness_function)

# Parameters
population_size = 10
num_generations = 20
lower_bound = 0
upper_bound = 31
mutation_rate = 0.1

# Run the Genetic Algorithm
best_solution = genetic_algorithm(population_size, num_generations, lower_bound, upper_bound, mutation_rate)
print(f"Best solution found: x = {best_solution}, f(x) = {fitness_function(best_solution)}")
