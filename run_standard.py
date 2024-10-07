import numpy as np
from helpers import *
from antq import AntQ, AntQGraph
from aco import ACO, ACOGraph

files = ['berlin52.csv','ch130.csv','kroa100.csv','oliver30.csv']
values = [52,130,100,30]

# a = read_coordinates_from_csv(f'dataset/standard/{files[00]}')
# print(len(a[0]))
# print(a[0])

for i in range(len(files)):
    cities = read_coordinates_from_csv(f'dataset/standard/{files[i]}')
    matrix = compute_distance_matrix(cities[0])
    
    antqg = AntQGraph(matrix)
    antq = AntQ(values[i],100,antqg)
    antq.run()

    print('erripuka')    
    print(f'antq best tour length: {antq.best_tour_len}')

    acog = ACOGraph(matrix)
    aco = ACO(values[i],100,acog)
    aco.run()
    print(f'aco best tour length: {aco.global_best_tour_len}')