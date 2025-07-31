import pandas as pd
import csv
import numpy as np
import random
import os

def read_dis(file_path):
    facility_costs = []
    distance_matrix = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter=',')
        first_row = next(reader)
        facility_costs = [float(cost) for cost in first_row if cost.strip() != '']
        for row in reader:
            distances = [float(dist) for dist in row if dist.strip() != '']
            if distances:
                distance_matrix.append(distances)
    transposed = [list(row) for row in zip(*distance_matrix)]
    return facility_costs, transposed

def generate_CL_to_file(k, percent, r, output_file):
    total_points = list(range(k))
    selected_num = int(k * percent / 100)
    selected_num = max(selected_num, 4)
    selected_points = random.sample(total_points, selected_num)
    random.shuffle(selected_points)

    CL = []
    i = 0
    with open(output_file, 'w') as f:
        while i < selected_num:
            remaining = selected_num - i
            if remaining < 2:
                break
            group_size = random.randint(2, min(r, remaining))
            group = selected_points[i:i + group_size]
            CL.append(group)
            f.write(str(group) + '\n')
            i += group_size

def read_facility_data(file_path, file_name, is_generate_CL = 0):
    os.makedirs(f'output\\{file_name}', exist_ok=True)
    facility_costs, distance_matrix = read_dis(file_path)
    k = len(distance_matrix[0])
    r = int(len(facility_costs)/4)
    if is_generate_CL:
        for percent in [10]:
            for i in range(1, 11):
                output_file = f'output\\{file_name}\\{percent}%CL_{i}.txt'
                generate_CL_to_file(k,percent,r,output_file)

    return facility_costs, distance_matrix


def read_CL(percent, i,file_name):
    file = f'output\\{file_name}\\{percent}%CL_{i}.txt'
    CL = []
    with open(file, 'r') as f:
        for line in f:
            pair = line.strip().strip('[]').split(',')
            CL.append([int(item) for item in pair])
    print(CL)
    return CL

def read_file(is_generate_CL,file_name):
    # file_path = f'ORLIB\\{file_name}.csv'
    file_path = f'M\\{file_name}.csv'
    facility_costs, distance_matrix = read_facility_data(file_path,file_name,is_generate_CL)
    return facility_costs, distance_matrix

if __name__ == "__main__":
    file_names = ['MP1','MP2','MP3','MP4','MP5']
    for file_name in file_names:
        read_file(1,file_name=file_name)