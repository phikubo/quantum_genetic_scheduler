#import
import os
import array
import random
import numpy
import numpy as np
#deap
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
#qiskit

from qiskit_optimization.applications import Knapsack             # clase principal para este tipo de problema
from qiskit.algorithms import QAOA, NumPyMinimumEigensolver
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.utils import algorithm_globals, QuantumInstance
from qiskit import Aer, BasicAer


def very_solution(this_indv):
    '''With rbs and received power, calculate modulation and finally tp.'''
    problem2 = Knapsack(values = received_power*np.array(this_indv.tolist()), weights = this_indv.tolist(), max_weight = capacity_rbs)
    #print('problem2: ',problem2)
    qp2 = problem2.to_quadratic_program()
    meo2 = MinimumEigenOptimizer(min_eigen_solver=NumPyMinimumEigensolver()) #coul it be out somewhere else?
    optimal_function_value2 = meo2.solve(qp2)
    solution=problem2.interpret(optimal_function_value2)
    return solution


def eval_function(individual):
    '''Fitness function'''
    #individual has the posible rbs combination
    #print("fit",len(individual.tolist()), len(received_power))
    problem = Knapsack(values = received_power*np.array(individual.tolist()), weights = individual.tolist(), max_weight = capacity_rbs)
    qp = problem.to_quadratic_program()
    seed = 123
    algorithm_globals.random_seed = seed
    qinstance = QuantumInstance(backend=Aer.get_backend('qasm_simulator'), shots=100,
                                seed_simulator=seed, seed_transpiler=seed)

    meo = MinimumEigenOptimizer(min_eigen_solver=QAOA(reps=1, quantum_instance=qinstance))
    optimal_function_value = meo.solve(qp)
    #xi=problem.interpret(optimal_function_value)
    print(individual, optimal_function_value.fval)
    return optimal_function_value.fval,

    #return np.sum(individual),

def ga_register(user_equipments, option_rbs):
    '''Structure of GA'''
    print("go ga!")

    #GA controls
    shuffle_mutate_prob=0.05
    cross_mate=0.8
    mutate_prob=0.7

    tournsizes=3
    populations=10
    generations=10
    #TOTAL ITERATIONS = tournsizes*populationsge*nerations

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", array.array, typecode='i', fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    # Attribute generator
    toolbox.register("attr_bool", random.choices, option_rbs, k=user_equipments)
    # Structure initializers
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_bool)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", eval_function)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=shuffle_mutate_prob)
    toolbox.register("select", tools.selTournament, tournsize=tournsizes)

    random.seed(64)
    pop = toolbox.population(n=populations)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=cross_mate, mutpb=mutate_prob, ngen=generations,
                                  stats=stats, halloffame=hof, verbose=False)

    return pop, log, hof

if __name__=="__main__":
    #Prototipo:
    print("Go main!")
    #lets asume we have 15 user terminals for which we have calculated the recived power
    global received_power
    received_power=np.array([-70, -50, -35, -45, -90, -38, -43, -83, -55, -68]) #in dBs
    received_power=100+received_power
    received_power=received_power[:4]
    #Each position represent the index of a user terminal, this means we have 10 user terminals
    user_equipments=len(received_power)
    #lets also asume our maximun numbers of chunks of frecuency are 1000
    global capacity_rbs
    capacity_rbs=2000
    option_rbs=range(capacity_rbs)
    pop, log, hof = ga_register(user_equipments, option_rbs)
    print("ga finished!")
    print("Best solution: {}\nTotal: {}, but max capacity is: {} rbs. \nPtx: {}".format(hof[0], np.sum(hof[0]), capacity_rbs, received_power))
    #to get the KP solution multiply hof[0] with xi. How to get xi out the function?
    #for time, we can recalculate it with KP.
    optimal_indexs=very_solution(hof[0])
    print("Filtered solution: ", optimal_indexs)
else:
    pass
    #print("Modulo Importado: [", os.path.basename(__file__), "]")
