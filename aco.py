import sys
import random
import math
from helpers import *

class ACOGraph:
    def __init__(self, dis_mat):
        self.matrix = dis_mat
        self.mat_size = len(dis_mat)
        self.pheromone = [[1 / (self.mat_size * self.mat_size) for _ in range(self.mat_size)] for _ in range(self.mat_size)]

class ACO:
    def __init__(self, number_of_ants, num_of_iteration, graph, alpha=1.0, beta=2.0, rho=0.5, q=100.0, strategy=2):
        self.graph = graph
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = q
        self.ant_count = number_of_ants
        self.generations = num_of_iteration
        self.strategy = strategy
        self.global_best_tour_len = sys.maxsize
        self.global_best_tour = []
        self.iter_best_tour = []
        self.iter_best_tour_len = sys.maxsize
        self.list_best_tour = []
        self.list_best_len = []
        self.list_avg = []
        self.list_var = []
        self.list_dev = []

    def _updatePheromone(self, ants):
        for i in range(self.graph.mat_size):
            for j in range(self.graph.mat_size):
                self.graph.pheromone[i][j] *= (1.0 - self.rho)
        
        for ant in ants:
            for i in range(len(ant.tabu) - 1):
                node_from = ant.tabu[i]
                node_to = ant.tabu[i + 1]
                if self.strategy == 1:
                    self.graph.pheromone[node_from][node_to] += self.Q
                elif self.strategy == 2:
                    self.graph.pheromone[node_from][node_to] += self.Q / self.graph.matrix[node_from][node_to]
                else:
                    self.graph.pheromone[node_from][node_to] += self.Q / ant.total_cost

    def iterRun(self):
        total_cost = 0
        best_cost = float('inf')
        best_solution = []

        ants = [_Ant(self, self.graph) for _ in range(self.ant_count)]

        for ant in ants:
            for _ in range(self.graph.mat_size - 1):
                ant._selectNext()
            ant.total_cost += self.graph.matrix[ant.tabu[-1]][ant.tabu[0]]
            total_cost += ant.total_cost

            if ant.total_cost < best_cost:
                best_cost = ant.total_cost
                best_solution = ant.tabu[:]
            
            ant._updatePheromoneDelta()

        avg_cost = total_cost / self.ant_count
        variance = sum((ant.total_cost - avg_cost) ** 2 for ant in ants) / (self.ant_count - 1)
        deviation = math.sqrt(variance)

        self._updatePheromone(ants)

        return best_solution, best_cost, avg_cost, variance, deviation

    def run(self):
        for gen in range(self.generations):
            iter_best_tour, iter_best_len, iter_avg, iter_variance, iter_deviation = self.iterRun()

            self.iter_best_tour_len = iter_best_len
            self.iter_best_tour = iter_best_tour[:]

            if self.global_best_tour_len > self.iter_best_tour_len:
                self.global_best_tour_len = self.iter_best_tour_len
                self.global_best_tour = self.iter_best_tour[:]

            self.list_best_tour.append(self.global_best_tour)
            self.list_best_len.append(self.global_best_tour_len)
            self.list_avg.append(iter_avg)
            self.list_var.append(iter_variance)
            self.list_dev.append(iter_deviation)

class _Ant:
    def __init__(self, aco: ACO, graph: ACOGraph):
        self.colony = aco
        self.graph = graph
        self.total_cost = 0.0
        self.tabu = []
        self.pheromone_delta = [[0 for _ in range(graph.mat_size)] for _ in range(graph.mat_size)]
        self.allowed = [i for i in range(graph.mat_size)]
        self.eta = [[0 if i == j else 1 / graph.matrix[i][j] for j in range(graph.mat_size)] for i in range(graph.mat_size)]
        start = random.randint(0, graph.mat_size - 1)
        self.tabu.append(start)
        self.current = start
        self.allowed.remove(start)

    def _selectNext(self):
        denominator = sum((self.graph.pheromone[self.current][i] ** self.colony.alpha) * (self.eta[self.current][i] ** self.colony.beta) for i in self.allowed)
        probabilities = [(self.graph.pheromone[self.current][i] ** self.colony.alpha) * (self.eta[self.current][i] ** self.colony.beta) / denominator for i in self.allowed]
        
        selected = random.choices(self.allowed, weights=probabilities)[0]
        
        self.allowed.remove(selected)
        self.tabu.append(selected)
        self.total_cost += self.graph.matrix[self.current][selected]
        self.current = selected

    def _updatePheromoneDelta(self):
        for i in range(1, len(self.tabu)):
            node_from = self.tabu[i - 1]
            node_to = self.tabu[i]
            if self.colony.strategy == 1:
                self.pheromone_delta[node_from][node_to] = self.colony.Q
            elif self.colony.strategy == 2:
                self.pheromone_delta[node_from][node_to] = self.colony.Q / self.graph.matrix[node_from][node_to]
            else:
                self.pheromone_delta[node_from][node_to] = self.colony.Q / self.total_cost

# num_cities = 10
# number_of_ants = 20
# num_of_iteration = 100
# alpha = 1.0
# beta = 2.0
# rho = 0.5
# q = 100.0
# strategy = 2

# coordinates = generate_random_coordinates(num_cities)
# distance_matrix = compute_distance_matrix(coordinates)

# graph = ACOGraph(distance_matrix)

# aco = ACO(number_of_ants, num_of_iteration, graph, alpha, beta, rho, q, strategy)

# aco.run()

# print("Best Tour Length:", aco.global_best_tour_len)
# print("Best Tour Path:", aco.global_best_tour)
