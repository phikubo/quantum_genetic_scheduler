#import
import os
import numpy as np
from deap import algorithms, base, creator, tools

def calculate_peso(rbs, modulacion):
	return rbs*modulacion

def main():
	'''main'''
	rbs=[10, 10, 20, 30]
	modulacion=[2, 4, 1, 3]
	pj=calculate_peso(rbs, modulacion)
	W=sum(rbs)

	#algoritmo genetico (algoritmo cuantico)




if __name__=="__main__":
	#Prototipo:
	main()
else:
	print("Modulo Importado: [", os.path.basename(__file__), "]")
