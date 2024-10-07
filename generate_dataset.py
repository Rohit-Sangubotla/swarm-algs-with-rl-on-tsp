import numpy as np
import csv

def generate_random_coordinates(num_cities):
    coordinates = np.random.rand(num_cities, 2) * 100
    return coordinates

def create_csv(num_cities, num_datasets, filename):
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['city_id', 'x', 'y']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for dataset in range(num_datasets):
            coordinates = generate_random_coordinates(num_cities)
            for i in range(num_cities):
                writer.writerow({'city_id': i+1, 'x': coordinates[i][0], 'y': coordinates[i][1]})
            writer.writerow({})

num_datasets = 20

city_sizes = [5, 25, 50, 100, 1000]
for num_cities in city_sizes:
    filename = f"cities_{num_cities}.csv"
    create_csv(num_cities, num_datasets, filename)
