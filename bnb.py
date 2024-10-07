import numpy as np
import sys

class TSPSolverBranchAndBound:
    def __init__(self, coordinates):
        self.coordinates = coordinates
        self.n = len(coordinates)
        self.graph = self.compute_distance_matrix(coordinates) 
        self.best_cost = float('inf')
        self.best_path = []
        self.visited = [False] * self.n
        self.final_res = float('inf')
        self.final_path = [None] * (self.n + 1)

    def compute_distance_matrix(self, coordinates):
        num_cities = len(coordinates)
        distance_matrix = np.zeros((num_cities, num_cities))
        for i in range(num_cities):
            for j in range(num_cities):
                distance_matrix[i][j] = np.linalg.norm(coordinates[i] - coordinates[j])
        return distance_matrix

    def copy_to_final(self, curr_path):
        self.final_path[:self.n + 1] = curr_path[:]
        self.final_path[self.n] = curr_path[0]

    def first_min(self, i):
        min_cost = sys.maxsize
        for k in range(self.n):
            if self.graph[i][k] < min_cost and i != k:
                min_cost = self.graph[i][k]
        return min_cost

    def second_min(self, i):
        first, second = sys.maxsize, sys.maxsize
        for j in range(self.n):
            if i == j:
                continue
            if self.graph[i][j] <= first:
                second = first
                first = self.graph[i][j]
            elif self.graph[i][j] <= second and self.graph[i][j] != first:
                second = self.graph[i][j]
        return second

    def tsp_recursive(self, curr_bound, curr_weight, level, curr_path):
        if level == self.n:
            if self.graph[curr_path[level - 1]][curr_path[0]] != 0:
                curr_res = curr_weight + self.graph[curr_path[level - 1]][curr_path[0]]
                if curr_res < self.final_res:
                    self.copy_to_final(curr_path)
                    self.final_res = curr_res
            return

        for i in range(self.n):
            if self.graph[curr_path[level - 1]][i] != 0 and not self.visited[i]:
                temp_bound = curr_bound
                curr_weight += self.graph[curr_path[level - 1]][i]
                if level == 1:
                    curr_bound -= ((self.first_min(curr_path[level - 1]) + self.first_min(i)) / 2)
                else:
                    curr_bound -= ((self.second_min(curr_path[level - 1]) + self.first_min(i)) / 2)

                if curr_bound + curr_weight < self.final_res:
                    curr_path[level] = i
                    self.visited[i] = True
                    self.tsp_recursive(curr_bound, curr_weight, level + 1, curr_path)

                curr_weight -= self.graph[curr_path[level - 1]][i]
                curr_bound = temp_bound
                self.visited = [False] * len(self.visited)
                for j in range(level):
                    if curr_path[j] != -1:
                        self.visited[curr_path[j]] = True

    def solve(self):
        curr_bound = 0
        curr_path = [-1] * (self.n + 1)
        for i in range(self.n):
            curr_bound += (self.first_min(i) + self.second_min(i))

        curr_bound = curr_bound // 2
        self.visited[0] = True
        curr_path[0] = 0

        self.tsp_recursive(curr_bound, 0, 1, curr_path)

        return self.final_res, self.final_path

def generate_random_coordinates(num_cities):
    coordinates = np.random.rand(num_cities, 2) * 100
    return coordinates

if __name__ == "__main__":
    coordinates = generate_random_coordinates(25)
    solver = TSPSolverBranchAndBound(coordinates)
    cost, path = solver.solve()
    print(f"Minimum cost: {cost}")
    print(f"Path: {path}")

# this is too slow and doesn't compute most of the time