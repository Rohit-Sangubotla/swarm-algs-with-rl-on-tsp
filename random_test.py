import numpy as np
import math
import random
import sys
from queue import Queue
from PyQt5.QtCore import QThread

# Generate random coordinates for cities
def generate_random_coordinates(num_cities):
    coordinates = np.random.rand(num_cities, 2) * 100  # Generate coordinates in the range [0, 100)
    return coordinates

# Compute the distance matrix based on coordinates
def compute_distance_matrix(coordinates):
    num_cities = len(coordinates)
    distance_matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            distance_matrix[i][j] = np.linalg.norm(coordinates[i] - coordinates[j])
    return distance_matrix


# Define the AntQGraph class
class AntQGraph:
    def __init__(self, dis_mat, aq_mat=None):
        self.aq_mat = aq_mat
        self.dis_mat = dis_mat
        if aq_mat is None:
            s = 0
            for i in range(0, len(self.dis_mat)):
                for j in range(0, len(self.dis_mat[i])):
                    if i != j:
                        s += dis_mat[i][j]
            self.aq_mat = [[(len(self.dis_mat) - 1) / s for _ in range(len(self.dis_mat))] for _ in range(len(self.dis_mat))]
        self.num_node = len(self.aq_mat)

    def getAntQValue(self, r, s):
        return self.aq_mat[r][s]

    def getDistance(self, r, s):
        return self.dis_mat[r][s]

    def getHeuristicValue(self, r, s):
        return 1.0 / self.getDistance(r, s)

    def getMaxAntQ(self, r):
        return max(self.aq_mat[r][:])

# Define the AntQ class
class AntQ(QThread):
    def __init__(self, number_of_ants, num_of_iteration, graph, alpha=.1, gamma=.3, delta=1, beta=2, w=10, q0=0.9, global_best=True, result=None):
        QThread.__init__(self)
        self.number_of_ants = number_of_ants
        self.alpha = alpha
        self.gamma = gamma
        self.delta = delta
        self.beta = beta
        self.graph = graph
        self.num_of_iteration = num_of_iteration
        self.w = w
        self.q0 = q0
        self.best_tours = []
        self.best_tour = []
        self.best_tour_len = sys.maxsize
        self.best_ant = -1
        self.ants = []
        self.global_best = global_best
        self.best_lens = []
        self.list_avg = []
        self.list_var = []
        self.list_dev = []
        self.result = result
        self.best_iter = 0

    def computeDelayValue(self):
        p_sum = 0
        for i in range(0, len(self.best_tour)):
            r = self.best_tour[i]
            if i < len(self.best_tour) - 1:
                s = self.best_tour[i + 1]
            else:
                s = self.best_tour[0]
            p_sum += self.graph.getDistance(r, s)
        if p_sum != 0:
            return self.w / p_sum
        else:
            return 0

    def updateDelayAntQ(self, tour):
        for i, node in enumerate(tour):
            r = node
            if i < len(tour) - 1:
                s = tour[i + 1]
            else:
                s = tour[0]
            ant_q_val = (1 - self.alpha) * self.graph.getAntQValue(r, s) + self.alpha * self.computeDelayValue()
            self.graph.aq_mat[r][s] = ant_q_val

    def createAnts(self):
        self.ants = []
        nodes = list(range(0, self.graph.num_node))
        starting_nodes = np.random.choice(nodes, self.number_of_ants, replace=True)
        for i in range(0, self.number_of_ants):
            ant = Ant(i, self, starting_nodes[i], self.q0)
            self.ants.append(ant)

    def computeIterTotal(self):
        return sum(ant.tour_len for ant in self.ants)

    def computeIterAvg(self):
        return self.computeIterTotal() / self.number_of_ants

    def computeLocalVariance(self, ant):
        return (ant.tour_len - self.computeIterAvg()) ** 2 / (self.number_of_ants - 1)

    def computeIterVariance(self):
        return sum(self.computeLocalVariance(ant) for ant in self.ants)

    def computeIterDeviation(self):
        variance = self.computeIterVariance()
        return math.sqrt(variance)

    def runIter(self, i):
        iter_min = sys.maxsize
        iter_best = []
        self.createAnts()
        for j in range(0, self.graph.num_node):
            for ant in self.ants:
                ant.move()

        for ant in self.ants:
            if ant.tour_len < self.best_tour_len:
                self.best_tour = ant.tour
                self.best_tour_len = ant.tour_len
                self.best_ant = ant.id
                self.best_iter = i

            if ant.tour_len < iter_min:
                iter_min = ant.tour_len
                iter_best = ant.tour

        iter_avg = self.computeIterAvg()
        iter_variance = self.computeIterVariance()

        return iter_avg, iter_variance, iter_best, iter_min

    def run(self):
        for i in range(0, self.num_of_iteration):
            iter_avg, iter_variance, iter_best, iter_min = self.runIter(i)
            iter_deviation = math.sqrt(iter_variance)

            if self.global_best:
                update_tour = self.best_tour
            else:
                update_tour = iter_best

            self.updateDelayAntQ(update_tour)

            aIter_result = {}
            aIter_result["iteration"] = i
            aIter_result["best_tour_len"] = self.best_tour_len
            aIter_result["best_tour"] = self.best_tour
            aIter_result["iter_avg"] = iter_avg
            aIter_result["iter_variance"] = iter_variance
            aIter_result["iter_deviation"] = iter_deviation
            if self.result is None:
                self.result = Queue()
            self.result.put(aIter_result)

            self.best_tours.append(self.best_tour)
            self.best_lens.append(self.best_tour_len)
            self.list_avg.append(iter_avg)
            self.list_var.append(iter_variance)
            self.list_dev.append(iter_deviation)

class Ant:
    def __init__(self, id, ant_q, start_node, q0=0.9):
        self.id = id
        self.start_node = start_node
        nodes_map = {}
        self.tour = [self.start_node]
        self.curr_node = start_node
        self.q0 = q0
        self.ant_q = ant_q
        self.tour_len = 0.0
        for i in range(0, self.ant_q.graph.num_node):
            if i != self.start_node:
                nodes_map[i] = i
        self.nodes_to_visit = list(nodes_map.values())

    def isEnd(self):
        return not self.nodes_to_visit

    def move(self):
        q = random.random()

        if not self.isEnd():
            max_node, max_val = self.getHeuristicMax()
            if q <= self.q0:
                next_node = max_node
            else:
                p = self.getNextNodesProbabilities()
                if not p:
                    p = [1.0 / len(self.nodes_to_visit)] * len(self.nodes_to_visit)

                next_node = np.random.choice(self.nodes_to_visit, 1, replace=False, p=p)[0]

            if next_node == -1:
                raise Exception("next_node < 0")

            self.nodes_to_visit.remove(next_node)

            learned_val = self.ant_q.graph.getMaxAntQ(next_node)
            self.updateAntQ(self.curr_node, next_node, learned_val)
            self.tour_len += self.ant_q.graph.getDistance(self.curr_node, next_node)
            self.tour.append(next_node)
            self.curr_node = next_node

        else:
            curr_node = self.tour[-1]
            next_node = self.tour[0]
            self.tour_len += self.ant_q.graph.getDistance(curr_node, next_node)

    def updateAntQ(self, curr_node, next_node, max_val):
        r = curr_node
        s = next_node
        alpha = self.ant_q.alpha
        gamma = self.ant_q.gamma
        graph = self.ant_q.graph
        ant_q_val = (1 - alpha) * graph.getAntQValue(r, s) + alpha * gamma * max_val
        graph.aq_mat[r][s] = ant_q_val

    def getNextNodesProbabilities(self):
        r = self.curr_node
        probabilities = []
        heu_sum = self.getHeuristicSum()
        if heu_sum != 0:
            for node in self.nodes_to_visit:
                p = self.getHeuristicValue(r, node) / heu_sum
                probabilities.append(p)
        return probabilities

    def getHeuristicValue(self, r, s):
        return math.pow(self.ant_q.graph.getAntQValue(r, s), self.ant_q.delta) * math.pow(self.ant_q.graph.getHeuristicValue(r, s), self.ant_q.beta)

    def getHeuristicMax(self):
        max_val = -1
        max_node = -1
        r = self.curr_node
        for s in self.nodes_to_visit:
            if self.getHeuristicValue(r, s) > max_val:
                max_val = self.getHeuristicValue(r, s)
                max_node = s
        return max_node, max_val

    def getHeuristicSum(self):
        h_sum = 0
        r = self.curr_node
        for s in self.nodes_to_visit:
            h_sum += self.getHeuristicValue(r, s)
        return h_sum

# Generate random examples of large TSP problems
def generate_and_solve_tsp(num_cities, num_ants, num_iterations, alpha, gamma, delta, beta, w, q0):
    coordinates = generate_random_coordinates(num_cities)
    distance_matrix = compute_distance_matrix(coordinates)
    graph = AntQGraph(dis_mat=distance_matrix)

    ant_q_algorithm = AntQ(
        number_of_ants=num_ants,
        num_of_iteration=num_iterations,
        graph=graph,
        alpha=alpha,
        gamma=gamma,
        delta=delta,
        beta=beta,
        w=w,
        q0=q0,
        global_best=True
    )

    ant_q_algorithm.run()

    best_tour = ant_q_algorithm.best_tour
    best_tour_length = ant_q_algorithm.best_tour_len

    print("Best Tour:", best_tour)
    print("Best Tour Length:", best_tour_length)

# Example usage
generate_and_solve_tsp(
    num_cities=50,  # Number of cities
    num_ants=50,    # Number of ants
    num_iterations=1000,  # Number of iterations
    alpha=0.1,
    gamma=0.3,
    delta=1,
    beta=2,
    w=10,
    q0=0.9
)