import numpy as np
import csv

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

def read_coordinates_from_csv(file_path):
    all_coordinates = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        coordinates = []
        next(reader)
        for row in reader:
            if row == ['', '', '']:  
                if coordinates:
                    all_coordinates.append(np.array(coordinates))
                    coordinates = []
            else:
                # print(row[1],"    " , row[2])
                coordinates.append([float(row[1]), float(row[2])])
        if coordinates:  
            all_coordinates.append(np.array(coordinates))
    return all_coordinates
