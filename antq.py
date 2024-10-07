import numpy as np
import sys
from queue import Queue
from helpers import *
import random

class AntQ:
    def __init__(self, number_of_ants, num_of_iteration, graph, alpha=.1, gamma=.3, delta=1, beta=2, w=10, q0=0.9, global_best=True, result=None):
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
        for i in range(len(self.best_tour)):
            r = self.best_tour[i]
            s = self.best_tour[(i + 1) % len(self.best_tour)]
            p_sum += self.graph.getDistance(r, s)
        return self.w / p_sum if p_sum != 0 else 0

    def updateDelayAntQ(self, tour):
        for i, node in enumerate(tour):
            r = node
            s = tour[(i + 1) % len(tour)]
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
        return np.sqrt(variance)

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

        return iter_avg, iter_variance, iter_best, iter_min

    def run(self):
        for i in range(self.num_of_iteration):
            iter_avg, iter_variance, iter_best, iter_min = self.runIter(i)
            iter_deviation = np.sqrt(iter_variance)

            update_tour = self.best_tour if self.global_best else iter_best
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
        self.tour = [self.start_node]
        self.curr_node = start_node
        self.q0 = q0
        self.ant_q = ant_q
        self.tour_len = 0.0
        self.nodes_to_visit = [i for i in range(ant_q.graph.num_node) if i != self.start_node]

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

                next_node = np.random.choice(self.nodes_to_visit, p=p)

            self.nodes_to_visit.remove(next_node)
            learned_val = self.ant_q.graph.getMaxAntQ(next_node)
            self.updateAntQ(self.curr_node, next_node, learned_val)
            self.tour_len += self.ant_q.graph.getDistance(self.curr_node, next_node)
            self.tour.append(next_node)
            self.curr_node = next_node
        else:
            self.tour_len += self.ant_q.graph.getDistance(self.tour[-1], self.tour[0])

    def updateAntQ(self, curr_node, next_node, max_val):
        r = curr_node
        s = next_node
        graph = self.ant_q.graph
        ant_q_val = (1 - self.ant_q.alpha) * graph.getAntQValue(r, s) + self.ant_q.alpha * self.ant_q.gamma * max_val
        graph.aq_mat[r][s] = ant_q_val

    def getNextNodesProbabilities(self):
        r = self.curr_node
        heu_sum = self.getHeuristicSum()
        if heu_sum == 0:
            return []
        return [self.getHeuristicValue(r, node) / heu_sum for node in self.nodes_to_visit]

    def getHeuristicValue(self, r, s):
        return (self.ant_q.graph.getAntQValue(r, s) ** self.ant_q.delta) * \
               (self.ant_q.graph.getHeuristicValue(r, s) ** self.ant_q.beta)

    def getHeuristicMax(self):
        r = self.curr_node
        max_node = max(self.nodes_to_visit, key=lambda s: self.getHeuristicValue(r, s))
        max_val = self.getHeuristicValue(r, max_node)
        return max_node, max_val

    def getHeuristicSum(self):
        r = self.curr_node
        return sum(self.getHeuristicValue(r, s) for s in self.nodes_to_visit)

class AntQGraph:
    def __init__(self, dis_mat, aq_mat=None):
        self.dis_mat = dis_mat
        if aq_mat is None:
            s = sum(dis_mat[i][j] for i in range(len(dis_mat)) for j in range(len(dis_mat[i])) if i != j)
            self.aq_mat = [[(len(dis_mat) - 1) / s for _ in range(len(dis_mat))] for _ in range(len(dis_mat))]
        else:
            self.aq_mat = aq_mat

        self.num_node = len(self.aq_mat)

    def getAntQValue(self, r, s):
        return self.aq_mat[r][s]

    def getDistance(self, r, s):
        return self.dis_mat[r][s]

    def getHeuristicValue(self, r, s):
        return 1.0 / self.getDistance(r, s)

    def getMaxAntQ(self, r):
        return max(self.aq_mat[r])

num_cities = 10
number_of_ants = 10
num_of_iteration = 100
coordinates = generate_random_coordinates(num_cities)
distance_matrix = compute_distance_matrix(coordinates)
graph = AntQGraph(distance_matrix)

ant_q = AntQ(number_of_ants=number_of_ants, num_of_iteration=num_of_iteration, graph=graph)

ant_q.run()

print("Best tour found:", ant_q.best_tour)
print("Best tour length:", ant_q.best_tour_len)
