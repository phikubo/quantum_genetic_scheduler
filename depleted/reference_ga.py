#https://github.com/DEAP/deap/blob/f4b77759897d0322ab5a6551106b28f6f4401a4e/examples/ga/onemax_short.py
import array
import random
import numpy
import numpy as np

from deap import algorithms
from deap import base
from deap import creator
from deap import tools


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", array.array, typecode='i', fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Attribute generator
capacity_rbs=1000
opciones_rbs=range(capacity_rbs)
user_equipments=15
toolbox.register("attr_bool", random.choices, opciones_rbs, k=user_equipments)

# Structure initializers
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_bool)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evalOneMax(individual):
    print(np.sum(individual), individual, type(individual))
    #help(individual)
    #En lugar de np.sum(individual), hacemos
    #KP(individual, tp(individual, modulacion), np.sum(max_prbs))
    return np.sum(individual),

toolbox.register("evaluate", evalOneMax)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    random.seed(64)

    pop = toolbox.population(n=5)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=10,
                                   stats=stats, halloffame=hof, verbose=False)

    return pop, log, hof

if __name__ == "__main__":
    pop, log, hof=main()
    ind = hof[0]
    print("Mejor combinacion: ")
    print(ind)
