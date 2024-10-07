import numpy as np
import csv
import sys
import random
import math
from queue import Queue
from helpers import *
from antq import AntQ, AntQGraph
from aco import ACO, ACOGraph

def solve_antq(file_path, num_ants, num_iterations, num):
    all_coordinates = read_coordinates_from_csv(file_path)
    results = []

    for coordinates in all_coordinates:
        distance_matrix = compute_distance_matrix(coordinates)
        graph = AntQGraph(dis_mat=distance_matrix)
        ant_q_algorithm = AntQ(
            number_of_ants=num_ants,
            num_of_iteration=num_iterations,
            graph=graph,
            coordinates=coordinates,
            alpha=0.1,
            gamma=0.3,
            delta=1,
            beta=2,
            w=10,
            q0=0.9,
            global_best=True
        )
        ant_q_algorithm.run()
        best_tour = ant_q_algorithm.best_tour
        best_tour_length = ant_q_algorithm.best_tour_len
        results.append((best_tour_length, best_tour))

    output_file = f'tsp_results_{num}.csv'
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['best_tour_length', 'best_tour'])
        for best_tour_length, best_tour in results:
            writer.writerow([best_tour_length, best_tour])
    print(f"Results saved to {output_file}")

def solve_aco(file_path, num_ants, num_iterations, num):
    all_coordinates = read_coordinates_from_csv(file_path)
    results = []

    for coordinates in all_coordinates:
        distance_matrix = compute_distance_matrix(coordinates)
        graph = ACOGraph(dis_mat=distance_matrix)
        aco_algorithm = ACO(
            number_of_ants=num_ants,
            num_of_iteration=num_iterations,
            graph=graph
        )
        aco_algorithm.run()
        best_tour = aco_algorithm.global_best_tour
        best_tour_length = aco_algorithm.global_best_tour_len
        results.append((best_tour_length, best_tour))

    output_file = f'aco_results_{num}.csv'
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['best_tour_length', 'best_tour'])
        for best_tour_length, best_tour in results:
            writer.writerow([best_tour_length, best_tour])
    print(f"Results saved to {output_file}")

cities = [5,25,50,100]

# for city in cities:
#     solve_aco(
#     file_path=f'cities_{city}.csv', 
#     num_ants=int(city),
#     num_iterations=100,
#     num = city
# )

for city in cities:
    solve_aco(
    file_path=f'cities_{city}.csv', 
    num_ants=int(city),
    num_iterations=100,
    num = city
)
