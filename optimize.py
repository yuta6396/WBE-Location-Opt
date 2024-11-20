import numpy as np
import random

from config import w_max, w_min, gene_length, crossover_rate, mutation_rate, alpha, tournament_size



#### ブラックボックス最適化手法 ####
###ランダムサーチ アルゴリズム  介入位置最適化問題
def random_search(objective_function, bounds, n_iterations, f_RS, num_input_grid, previous_best=None):
    # 以前の最良のスコアとパラメータを初期化
    input_history=[]
    if previous_best is None:
        best_score = float('inf')
        best_params = None
    else:
        best_params, best_score = previous_best
    for _ in range(n_iterations):
        candidate = [np.random.randint(bound.low, bound.high + 1) for bound in bounds]
        input_history.append(candidate)
        score = objective_function(candidate)
        if score < best_score:
            best_score = score
            best_params = candidate
    f_RS.write(f"\n input_history \n{input_history}")

    return best_params, best_score

def random_search_YZI(objective_function, bounds, bounds_input, n_iterations, f_RS, num_input_grid, previous_best=None):
    # 以前の最良のスコアとパラメータを初期化
    input_history=[]
    if previous_best is None:
        best_score = float('inf')
        best_params = None
    else:
        best_params, best_score = previous_best
    for _ in range(n_iterations):
        candidate =[]
        candidate.append(np.random.randint(bounds[0].low, bounds[0].high + 1))
        candidate.append(np.random.randint(bounds[1].low, bounds[1].high + 1))
        candidate.append(np.random.uniform(bounds_input[0][0], bounds_input[0][1]))
        input_history.append(candidate)
        score = objective_function(candidate)
        if score < best_score:
            best_score = score
            best_params = candidate
    f_RS.write(f"\n input_history \n{input_history}")

    return best_params, best_score
###PSO アルゴリズム

# 粒子の初期化
def initialize_particles(num_particles, bounds):
    particles = []
    for _ in range(num_particles):
        # position = np.array([np.random.randint(bound, bound[1]) for bound in bounds])
        position = [np.random.randint(bound.low, bound.high + 1) for bound in bounds]
        velocity = np.array([np.random.uniform(-1, 1) for _ in bounds])
        particles.append({
            'position': position,
            'velocity': velocity,
            'best_position': position.copy(),
            'best_value': float('inf'),
            'value': float('inf')
        })
    return particles

# 速度の更新
def update_velocity(particle, global_best_position, w, c1, c2):
    r1 = np.random.random(len(particle['position']))
    r2 = np.random.random(len(particle['position']))
    cognitive = c1 * r1 * (particle['best_position'] - particle['position'])
    social = c2 * r2 * (global_best_position - particle['position'])
    particle['velocity'] = w * particle['velocity'] + cognitive + social

# 位置の更新
def update_position(particle, bounds):
    particle['position'] += particle['velocity']
    # for i in range(len(particle['position'])):
    #     if particle['position'][i] < bounds[i][0]:
    #         particle['position'][i] = bounds[i][0]
    #     if particle['position'][i] > bounds[i][1]:
    #         particle['position'][i] = bounds[i][1]
    print(particle['position'])
    for i, bound in enumerate(bounds):
        if particle['position'][i] < bound.low:
            particle['position'][i] = bound.low
        if particle['position'][i] > bound.high:
            particle['position'][i] = bound.high

# PSOアルゴリズムの実装
def PSO(objective_function, bounds, num_particles, num_iterations, f_PSO):
    particles = initialize_particles(num_particles, bounds)
    global_best_value = float('inf')
    #global_best_position = np.array([np.random.uniform(bound[0], bound[1]) for bound in bounds])
    global_best_position = [np.random.randint(bound.low, bound.high + 1) for bound in bounds]
    w = w_max      # 慣性係数
    c1 = 2.0      # 認知係数
    c2 = 2.0      # 社会係数

    result_value = np.zeros(num_iterations)
    flag_b = 0
    for iteration in range(num_iterations):
        w = w_max - (w_max - w_min)*(iteration+1)/(num_iterations)
        flag_s = 0
        f_PSO.write(f'w={w}')
        print(f'w={w}')
        for particle in particles:
            rounded_particle_position = np.round(particle['position'])
            particle['value'] = objective_function(rounded_particle_position)
            print(particle['value'])
            if particle['value'] < particle['best_value']:
                particle['best_value'] = rounded_particle_position
                particle['best_position'] = rounded_particle_position.copy()

            if particle['value'] < global_best_value:
                global_best_value = particle['value']
                global_best_position = rounded_particle_position.copy()

            if flag_s == 0 :
                iteration_positions = particle['position'].copy()  ##ここだけ動きを見たいからあえてこうしている
                flag_s = 1
            else:
                iteration_positions = np.vstack((iteration_positions, particle['position'].copy()))
        if flag_b == 0:
            position_history = iteration_positions
            flag_b = 1
        else:
            position_history = np.vstack(((position_history, iteration_positions)))# イテレーションごとの位置を保存

        for particle in particles:
            update_velocity(particle, global_best_position, w, c1, c2)
            update_position(particle, bounds)

        result_value[iteration] = global_best_value
        print(f"Iteration {iteration + 1}/{num_iterations}, Best Value: {global_best_value}")
        
    formatted_data = '[' + ',\n '.join([str(list(row)) for row in position_history]) + ']'
    f_PSO.write(f"\nposition_history={formatted_data}")
    return global_best_position,  result_value


###遺伝的アルゴリズム
def initialize_population(pop_size, gene_length, lower_bound, upper_bound):
    return np.random.uniform(lower_bound, upper_bound, (pop_size, gene_length))


def calculate_fitness(population, fitness_function):
    # Apply the fitness function to each individual in the population
    return np.apply_along_axis(fitness_function, 1, population)


def tournament_selection(population, fitness, tournament_size):
    selected_parents = []
    for _ in range(len(population)):
        participants_idx = np.random.choice(np.arange(len(population)), tournament_size, replace=False)
        best_idx = participants_idx[np.argmin(fitness[participants_idx])]
        selected_parents.append(population[best_idx])
    return np.array(selected_parents)


def blx_alpha_crossover(parents, offspring_size, alpha):
    offspring = np.empty(offspring_size)
    for i in range(0, offspring_size[0], 2):
        parent1_idx = i % parents.shape[0]
        parent2_idx = (i + 1) % parents.shape[0]
        
        parent1 = parents[parent1_idx]
        parent2 = parents[parent2_idx]
        
        min_gene = np.minimum(parent1, parent2)
        max_gene = np.maximum(parent1, parent2)
        
        diff = max_gene - min_gene
        lower_bound = min_gene - alpha * diff
        upper_bound = max_gene + alpha * diff
        
        offspring[i] = np.random.uniform(lower_bound, upper_bound)
        if i + 1 < offspring_size[0]:
            offspring[i + 1] = np.random.uniform(lower_bound, upper_bound)
    
    return offspring


def mutate(offspring, mutation_rate, lower_bound, upper_bound):
    for idx in range(offspring.shape[0]):
        for gene_idx in range(offspring.shape[1]):
            if np.random.rand() < mutation_rate:
                offspring[idx, gene_idx] = np.random.uniform(lower_bound, upper_bound)
        # Clamp values to be within the bounds using np.clip
        offspring[idx] = np.clip(offspring[idx], lower_bound, upper_bound)
    return offspring



def genetic_algorithm(objective_function, pop_size, gene_length, num_generations, crossover_rate,
                      mutation_rate, lower_bound, upper_bound, alpha, tournament_size, f_GA):
    best_fitness = float("inf")
    best_individual = None

    population = initialize_population(pop_size, gene_length, lower_bound, upper_bound)
    
    gene_history=population.copy() # 粒子の現在位置を記録     粒子数*num_grid

    for generation in range(num_generations):
        fitness = calculate_fitness(population, objective_function)

        current_best_fitness = np.min(fitness)
        current_best_individual = population[np.argmin(fitness)]

        if current_best_fitness < best_fitness:
            f_GA.write(f"{current_best_fitness=},  {best_fitness=}")
            best_fitness = current_best_fitness.copy()
            best_individual = current_best_individual.copy()


        print(f"Generation {generation + 1}: Best Fitness = {best_fitness}, Best Individual = {best_individual}")
        f_GA.write(f"\nGeneration {generation + 1}: Best Fitness = {best_fitness}, Best Individual = {best_individual}")
        parents = tournament_selection(population, fitness, tournament_size)

        offspring_size = (int(pop_size * crossover_rate), gene_length)
        offspring = blx_alpha_crossover(parents, offspring_size, alpha)

        offspring = mutate(offspring, mutation_rate, lower_bound, upper_bound)
        
        population[0:offspring.shape[0]] = offspring
        if generation < num_generations - 1:
            gene_history = np.vstack((gene_history, population.copy()))  # 粒子の現在位置を記録

    f_GA.write(f"\n{gene_history=}")
    return best_fitness, best_individual