import sys
from evoman.environment import Environment
from demo_controller import player_controller

# imports other libs
import time
import numpy as np
import glob, os

n_hidden_neurons = 10
dom_u = 1
dom_l = -1
npop = 500
gens = 10
mutation = 0.3
last_best = 0
elite_size = 0.05  # Retain the top 5% individuals as elites

run_mode = ('train')

experiment_name = 'Elitism_test'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

env = Environment(experiment_name=experiment_name,
                  enemies=[8],
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  visuals=True)

env.state_to_log()
n_inputs = env.get_num_sensors()  # Get the number of sensors (input neurons)
print(f"Number of input neurons (sensors): {n_inputs}")
n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

ini = time.time()

def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f

def norm(x, pfit_pop):
    if ( max(pfit_pop) - min(pfit_pop) ) > 0:
        x_norm = ( x - min(pfit_pop) )/( max(pfit_pop) - min(pfit_pop) )
    else:
        x_norm = 0

    if x_norm <= 0:
        x_norm = 0.0000000001
    return x_norm

# Finds the fitness score of each individual
def evaluate(x):
    return np.array(list(map(lambda y: simulation(env,y), x)))

def tournament(pop):
    c1 = np.random.randint(0, pop.shape[0], 1)
    c2 = np.random.randint(0, pop.shape[0], 1)

    if fit_pop[c1] > fit_pop[c2]:
        return pop[c1][0]
    else:
        return pop[c2][0]

def limits(x):
    if x > dom_u:
        return dom_u
    elif x < dom_l:
        return dom_l
    else:
        return x

# Creates offspring and mutates them.
# Returns a new population of offspring in vectors.
def crossover(pop):
    total_offspring = np.zeros((0, n_vars))

    for p in range(0, pop.shape[0], 2):
        p1 = tournament(pop)
        p2 = tournament(pop)

        n_offspring = np.random.randint(1, 3 + 1, 1)[0]
        offspring = np.zeros((n_offspring, n_vars))

        for f in range(0, n_offspring):
            cross_prop = np.random.uniform(0, 1)
            offspring[f] = p1 * cross_prop + p2 * (1 - cross_prop)

            # mutation
            for i in range(0, len(offspring[f])):
                if np.random.uniform(0, 1) <= mutation:
                    offspring[f][i] = offspring[f][i] + np.random.normal(0, 1)

            offspring[f] = np.array(list(map(lambda y: limits(y), offspring[f])))

            total_offspring = np.vstack((total_offspring, offspring[f]))

    return total_offspring

# Elitism selection method
def elitist_selection(fit_pop, pop, elite_size=0.05):
    num_elites = int(elite_size * npop)
    sorted_indices = np.argsort(fit_pop)[::-1]
    elites = pop[sorted_indices[:num_elites]]  # Select top individuals

    fit_pop_cp = fit_pop
    fit_pop_norm = np.array(list(map(lambda y: norm(y, fit_pop_cp), fit_pop)))  # Normalize fitness values
    probs = (fit_pop_norm) / (fit_pop_norm).sum()  # Compute selection probabilities
    chosen = np.random.choice(pop.shape[0], npop - num_elites, p=probs, replace=False)  # Select the rest

    new_population = np.vstack((elites, pop[chosen]))
    new_fit_pop = np.append(fit_pop[sorted_indices[:num_elites]], fit_pop[chosen])

    return new_population, new_fit_pop

if run_mode == 'test':
    bsol = np.loadtxt(experiment_name+'/best.txt')
    print( '\n RUNNING SAVED BEST SOLUTION \n')
    env.update_parameter('speed','normal')
    evaluate([bsol])

    sys.exit(0)

if not os.path.exists(experiment_name+'/evoman_solstate'):
    print( '\nNEW EVOLUTION\n')

    pop = np.random.uniform(dom_l, dom_u, (npop, n_vars))
    fit_pop = evaluate(pop)
    best = np.argmax(fit_pop)
    mean = np.mean(fit_pop)
    std = np.std(fit_pop)
    ini_g = 0
    solutions = [pop, fit_pop]
    env.update_solutions(solutions)

else:
    print( '\nCONTINUING EVOLUTION\n')
    env.load_state()
    pop = env.solutions[0]
    fit_pop = env.solutions[1]

    best = np.argmax(fit_pop)
    mean = np.mean(fit_pop)
    std = np.std(fit_pop)

    file_aux = open(experiment_name+'/gen.txt', 'r')
    ini_g = int(file_aux.readline())
    file_aux.close()

file_aux = open(experiment_name+'/results.txt', 'a')
file_aux.write('\n\ngen best mean std')
print('\n GENERATION ' + str(ini_g) + ' ' + str(round(fit_pop[best], 6)) + ' ' + str(round(mean, 6)) + ' ' + str(round(std, 6)))
file_aux.write('\n' + str(ini_g) + ' ' + str(round(fit_pop[best], 6)) + ' ' + str(round(mean, 6)) + ' ' + str(round(std, 6)))
file_aux.close()

last_sol = fit_pop[best]
notimproved = 0

for i in range(ini_g + 1, gens):

    offspring = crossover(pop)  # Crossover
    fit_offspring = evaluate(offspring)  # Evaluate the offspring
    pop = np.vstack((pop, offspring))
    fit_pop = np.append(fit_pop, fit_offspring)

    # Apply elitist selection
    pop, fit_pop = elitist_selection(fit_pop, pop, elite_size=0.05)

    best = np.argmax(fit_pop)  # Find the best solution
    fit_pop[best] = float(evaluate(np.array([pop[best]]))[0])  # Recheck best fitness for stability
    best_sol = fit_pop[best]

    if best_sol <= last_sol:
        notimproved += 1
    else:
        last_sol = best_sol
        notimproved = 0

    if notimproved >= 15:
        file_aux = open(experiment_name+'/results.txt', 'a')
        file_aux.write('\ndoomsday')
        file_aux.close()

        pop, fit_pop = doomsday(pop, fit_pop)
        notimproved = 0

    best = np.argmax(fit_pop)
    std = np.std(fit_pop)
    mean = np.mean(fit_pop)

    # Save results
    file_aux = open(experiment_name+'/results.txt', 'a')
    print('\n GENERATION ' + str(i) + ' ' + str(round(fit_pop[best], 6)) + ' ' + str(round(mean, 6)) + ' ' + str(round(std, 6)))
    file_aux.write('\n' + str(i) + ' ' + str(round(fit_pop[best], 6)) + ' ' + str(round(mean, 6)) + ' ' + str(round(std, 6)))
    file_aux.close()

    file_aux = open(experiment_name+'/gen.txt', 'w')
    file_aux.write(str(i))
    file_aux.close()

    np.savetxt(experiment_name+'/best.txt', pop[best])  # Save the best solution

    solutions = [pop, fit_pop]
    env.update_solutions(solutions)
    env.save_state()

fim = time.time()  # Print total execution time for experiment
print('\nExecution time: ' + str(round((fim - ini) / 60)) + ' minutes \n')
print('\nExecution time: ' + str(round((fim - ini))) + ' seconds \n')

file = open(experiment_name+'/neuroended', 'w')  # Save control (simulation has ended) file for bash loop file
file.close()


env.state_to_log()
