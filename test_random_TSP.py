import numpy as np
import matplotlib.pyplot as plt
import random
import math
import sys
from queue import Queue

def generate_random_coordinates(num_cities):
    coordinates = np.random.rand(num_cities, 2) * 100
    return coordinates

def compute_distance_matrix(coordinates):
    num_cities = len(coordinates)
    distance_matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            distance_matrix[i][j] = np.linalg.norm(coordinates[i] - coordinates[j])
    return distance_matrix

class AntQGraph:
    def __init__(self, dis_mat, aq_mat=None):
        self.aq_mat = aq_mat
        self.dis_mat = dis_mat
        if aq_mat is None:
            s = 0
            for i in range(len(self.dis_mat)):
                for j in range(len(self.dis_mat[i])):
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

class AntQ:
    def __init__(self, number_of_ants, num_of_iteration, graph, coordinates, alpha=.1, gamma=.3, delta=1, beta=2, w=10, q0=0.9, global_best=True, result=None):
        self.number_of_ants = number_of_ants
        self.alpha = alpha
        self.gamma = gamma
        self.delta = delta
        self.beta = beta
        self.graph = graph
        self.coordinates = coordinates 
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
        for i in range(len(self.best_tour)):
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
        nodes = list(range(self.graph.num_node))
        starting_nodes = np.random.choice(nodes, self.number_of_ants, replace=True)
        for i in range(self.number_of_ants):
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
        return np.math.sqrt(variance)

    def visualize_intermediate_steps(self, iteration):
        if iteration % (self.num_of_iteration // 5) == 0:
            plt.figure(figsize=(10, 6))
            for ant in self.ants:
                tour_coords = np.array([self.coordinates[i] for i in ant.tour + [ant.tour[0]]])
                plt.plot(tour_coords[:, 0], tour_coords[:, 1], alpha=0.3)
            plt.scatter(self.coordinates[:, 0], self.coordinates[:, 1], color='red', label='Cities')
            plt.title(f'Iteration {iteration}')
            plt.show()

    def runIter(self, i):
        iter_min = sys.maxsize
        iter_best = []
        self.createAnts()
        for j in range(self.graph.num_node):
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

        self.visualize_intermediate_steps(i)

        return iter_avg, iter_variance, iter_best, iter_min

    def run(self):
        for i in range(self.num_of_iteration):
            iter_avg, iter_variance, iter_best, iter_min = self.runIter(i)
            iter_deviation = np.math.sqrt(iter_variance)

            if self.global_best:
                update_tour = self.best_tour
            else:
                update_tour = iter_best

            self.updateDelayAntQ(update_tour)

            aIter_result = {
                "iteration": i,
                "best_tour_len": self.best_tour_len,
                "best_tour": self.best_tour,
                "iter_avg": iter_avg,
                "iter_variance": iter_variance,
                "iter_deviation": iter_deviation,
            }
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
        for i in range(self.ant_q.graph.num_node):
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

def plot_tsp(coordinates, best_tour):
    plt.figure(figsize=(10, 6))
    plt.scatter(coordinates[:, 0], coordinates[:, 1], color='red', label='Cities')

    for i, txt in enumerate(range(len(coordinates))):
        plt.annotate(txt, (coordinates[i, 0], coordinates[i, 1]))

    tour_coords = np.array([coordinates[i] for i in best_tour + [best_tour[0]]])
    plt.plot(tour_coords[:, 0], tour_coords[:, 1], color='blue', label='Tour')

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Traveling Salesman Problem Solution')
    plt.legend()
    plt.show()

def generate_and_solve_tsp(num_cities, num_ants, num_iterations, alpha, gamma, delta, beta, w, q0):
    coordinates = generate_random_coordinates(num_cities)
    distance_matrix = compute_distance_matrix(coordinates)
    graph = AntQGraph(dis_mat=distance_matrix)

    plot_tsp(coordinates, list(range(num_cities)))

    ant_q_algorithm = AntQ(
        number_of_ants=num_ants,
        num_of_iteration=num_iterations,
        graph=graph,
        coordinates=coordinates,
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

    plot_tsp(coordinates, best_tour)

generate_and_solve_tsp(
    num_cities=100,  
    num_ants=100,  
    num_iterations=100, 
    alpha=0.1,
    gamma=0.3,
    delta=1,
    beta=2,
    w=10,
    q0=0.9
)
