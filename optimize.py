import numpy as np
import random
from deap import base, creator, tools, algorithms
from sklearn.utils import check_random_state

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
    best_position = np.array(particle['best_position'])
    position = np.array(particle['position'])
    cognitive = c1 * r1 * (best_position - position)
    social = c2 * r2 * (global_best_position - position)
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
            # print("AA")
            print(particle['position'])
            # print("AA")
            #rounded_particle_position = np.round(particle['position'])
            particle['value'] = objective_function(np.rint(particle['position']).astype(int))
            if particle['value'] < particle['best_value']:
                particle['best_value'] = particle['value']
                particle['best_position'] = particle['position'].copy()

            if particle['value'] < global_best_value:
                global_best_value = particle['value']
                global_best_position = np.rint(particle['position']).astype(int)

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
# def initialize_population(pop_size, gene_length, bounds):
#     return np.random.uniform(lower_bound, upper_bound, (pop_size, gene_length))
# def initialize_population(pop_size, gene_length, bounds):
#     random_samples2 = np.random.randint(0, 39, (pop_size, 1))
#     random_samples3 = np.random.randint(0, 96, (pop_size, 1))
#     combined_samples = np.concatenate((random_samples2,random_samples3), axis=1)
#     return combined_samples

# def calculate_fitness(population, fitness_function):
#     # Apply the fitness function to each individual in the population
#     return np.apply_along_axis(fitness_function, 1, population)


# def tournament_selection(population, fitness, tournament_size):
#     selected_parents = []
#     for _ in range(len(population)):
#         participants_idx = np.random.choice(np.arange(len(population)), tournament_size, replace=False)
#         best_idx = participants_idx[np.argmin(fitness[participants_idx])]
#         selected_parents.append(population[best_idx])
#     return np.array(selected_parents)


# def blx_alpha_crossover(parents, offspring_size, alpha):
#     offspring = np.empty(offspring_size)
#     for i in range(0, offspring_size[0], 2):
#         parent1_idx = i % parents.shape[0]
#         parent2_idx = (i + 1) % parents.shape[0]
        
#         parent1 = parents[parent1_idx]
#         parent2 = parents[parent2_idx]
        
#         min_gene = np.minimum(parent1, parent2)
#         max_gene = np.maximum(parent1, parent2)
        
#         diff = max_gene - min_gene
#         lower_bound = min_gene - alpha * diff
#         upper_bound = max_gene + alpha * diff
        
#         offspring[i] = np.random.uniform(lower_bound, upper_bound)
#         if i + 1 < offspring_size[0]:
#             offspring[i + 1] = np.random.uniform(lower_bound, upper_bound)
    
#     return offspring


# def mutate(offspring, mutation_rate, bounds):
#     for idx in range(offspring.shape[0]):
#         for gene_idx in range(offspring.shape[1]):
#             if np.random.rand() < mutation_rate:
#                 offspring[idx, gene_idx] = np.random.randint(bound.low, bound.high最大化問題最大化問題
#         # Clamp values to be within the bounds using np.clip
#         offspring[idx] = np.clip(offspring[idx], lower_bound, upper_bound)
#     return offspring



# def genetic_algorithm(objective_function, pop_size, gene_length, num_generations, crossover_rate,
#                       mutation_rate, bounds, alpha, tournament_size, f_GA):
#     best_fitness = float("inf")
#     best_individual = None

#     population = initialize_population(pop_size, gene_length, bounds)
    
#     gene_history=population.copy() # 粒子の現在位置を記録     粒子数*num_grid

#     for generation in range(num_generations):
#         fitness = calculate_fitness(population, objective_function)

#         current_best_fitness = np.min(fitness)
#         current_best_individual = population[np.argmin(fitness)]

#         if current_best_fitness < best_fitness:
#             f_GA.write(f"{current_best_fitness=},  {best_fitness=}")
#             best_fitness = current_best_fitness.copy()
#             best_individual = current_best_individual.copy()


#         print(f"Generation {generation + 1}: Best Fitness = {best_fitness}, Best Individual = {best_individual}")
#         f_GA.write(f"\nGeneration {generation + 1}: Best Fitness = {best_fitness}, Best Individual = {best_individual}")
#         parents = tournament_selection(population, fitness, tournament_size)

#         offspring_size = (int(pop_size * crossover_rate), gene_length)
#         offspring = blx_alpha_crossover(parents, offspring_size, alpha)

#         offspring = mutate(offspring, mutation_rate, bounds)
        
#         population[0:offspring.shape[0]] = offspring
#         if generation < num_generations - 1:
#             gene_history = np.vstack((gene_history, population.copy()))  # 粒子の現在位置を記録

#     f_GA.write(f"\n{gene_history=}")
#     return best_fitness, best_individual

# 遺伝的アルゴリズムのパラメータ設定
LOWER_BOUNDS = [0, 0]  # 各次元の下限
UPPER_BOUNDS = [39, 96]# 各次元の上限



def genetic_algorithm(objective_function, pop_size, gene_length, num_generations, crossover_rate,
                       mutation_rate, alpha, tournament_size, f_GA):
        # creatorの再定義を防ぐ
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,)) # weights=(1.0,)なら最大化問題
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMin)

    # 基本設定を行います
    toolbox = base.Toolbox()
    random_state = check_random_state(None)

    # 各次元の値を生成する関数を登録
    for i in range(len(LOWER_BOUNDS)):
        toolbox.register(f"attr_int_{i}", random_state.randint, LOWER_BOUNDS[i], UPPER_BOUNDS[i] + 1)

    # 個体の作成方法を登録
    toolbox.register("individual", tools.initCycle, creator.Individual,
                    [getattr(toolbox, f"attr_int_{i}") for i in range(len(LOWER_BOUNDS))], n=1)
    # 集団の作成方法を登録
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # 評価関数を登録
    toolbox.register("evaluate", objective_function)
    # 交叉方法を登録
    toolbox.register("mate", tools.cxTwoPoint) #2点交叉

    # 突然変異方法を登録（範囲内で突然変異を行います）
    def mutate(individual):
        i = random_state.randint(0, len(individual))
        individual[i] = random_state.randint(LOWER_BOUNDS[i], UPPER_BOUNDS[i] + 1)
        return (individual,)

    toolbox.register("mutate", mutate)
    # 個体選択方法を登録
    toolbox.register("select", tools.selTournament, tournsize=tournament_size)

    # 初期集団の作成
    population = toolbox.population(n=pop_size)

    # 遺伝的アルゴリズムの実行
    for generation in range(num_generations):
        f_GA.write(f'Generation {generation + 1}\n')
        offspring = algorithms.varAnd(population, toolbox, crossover_rate, mutation_rate)
        fits = list(map(toolbox.evaluate, offspring))

        for fit, ind in zip(fits, offspring):
            if not isinstance(fit, (list, tuple)):
                fit = (fit,)
            ind.fitness.values = fit
            f_GA.write(f'Individual: {ind}, Fitness: {fit[0]}\n')

        population = toolbox.select(offspring, k=len(population))

    best_individual = tools.selBest(population, k=1)[0]
    return best_individual.fitness.values[0], best_individual