#import
import os
import numpy as np
import array
import random
from deap import algorithms, base, creator, tools

def calculate_peso():
    rbs=np.array([10, 10, 20, 30])
    modulacion=np.array([2, 4, 1, 3])
    return rbs*modulacion

def eval_throguthput(individual):
    '''evaluate fitness of tp'''
    recursos=np.sum(individual)
    print("prbs: ",recursos, individual)
    pj=calculate_peso()
    #pj=sum(simple_throughput_pj)
    pj=np.sum(pj)
    #ESPACIO PARA EL ALGORITMO CUANTICO.
    if pj>5000:
        pj=pj*1.1
    else:
        pj=pj*0.1
    return pj,


def genetic_algorithm(user_equipments, wanted_generations, indpb, tournsize, opciones_rbs):
    mating_prob=0.7
    mutating_prob=0.2
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", array.array, typecode='i', fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    toolbox.register("indices", random.choices, opciones_rbs, k=user_equipments)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxOrdered)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=indpb) #antes 0.05
    toolbox.register("select", tools.selTournament, tournsize=tournsize)
    #
    toolbox.register("evaluate", eval_throguthput)
    #start with a population of 300 individuals
    pop = toolbox.population(n=wanted_generations)
    #only save the very best one
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    # use one of the built in GA's with a probablilty of mating of 0.7
    # a probability of mutating 0.2 and 140 generations.
    algorithms.eaSimple(pop, toolbox, mating_prob, mutating_prob, 200, stats=stats, halloffame=hof) #antes 0.1
    return pop, stats, hof

def main():
    '''main'''
    #related to simulator

    #rbs=np.array([10, 10, 20, 30])
    capacity_rbs=3000
    opciones_rbs=range(capacity_rbs)

    potencia_rec=np.array([3,4,5,6])
    #modulacion=np.array([2, 4, 1, 3])
    user_equipments=len(potencia_rec)
    #related to GAs.
    wanted_generations=10
    indpb=0.09
    tournsize=5

    pop, stats, hof=genetic_algorithm(user_equipments, wanted_generations, indpb, tournsize, opciones_rbs)

    #algoritmo genetico (algoritmo cuantico)




if __name__=="__main__":
    #Prototipo:
    main()
else:
    print("Modulo Importado: [", os.path.basename(__file__), "]")
