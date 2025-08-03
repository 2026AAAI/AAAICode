import copy
import time
import data_deal_FL
import numpy as np
import evaluate
from scipy.optimize import linear_sum_assignment

class Greedy():
    def __init__(self, distance_matrix, Y, facility_costs):
        self.cij = distance_matrix
        self.client_num = len(distance_matrix[0])
        self.facility_num = len(facility_costs)
        self.CL = Y
        self.F = set(range(0, self.facility_num))
        self.X = set()
        self.P = set(range(0, self.client_num))
        self.S = set(range(0, self.client_num))
        self.assign_client = [-1] * self.client_num
        self.facility_costs = copy.deepcopy(facility_costs)
        self.open_num = 0

    def open_facility(self):
        while self.S:
            best_ratio = float('inf')
            best_i = None
            best_Y = set()

            for i in self.F:
                savings = 0
                for j in self.P - self.S:
                    current_cost = min(self.cij[k][j] for k in self.X)
                    if current_cost > self.cij[i][j]:
                        savings += (current_cost - self.cij[i][j])

                sorted_clients = sorted(self.S, key=lambda j: self.cij[i][j])
                prefix_cost = 0
                prev_ratio = float('inf')

                for idx, j in enumerate(sorted_clients, 1):
                    prefix_cost += self.cij[i][j]
                    ratio = (self.facility_costs[i] - savings + prefix_cost) / idx

                    if ratio > prev_ratio:
                        break
                    else:
                        if ratio < best_ratio:
                            best_ratio = ratio
                            best_i = i
                            best_Y = set(sorted_clients[:idx])
                    prev_ratio = ratio

            for j in best_Y:
                self.assign_client[j] = best_i

            for j in self.P-self.S:
                o = self.assign_client[j]
                if self.cij[best_i][j] < self.cij[o][j]:
                    self.assign_client[j] = best_i

            self.facility_costs[best_i] = 0
            self.X.add(best_i)
            self.S -= best_Y

    def open_facility_for_CL(self):
        self.open_num = len(self.X)
        max_CL_size = len(max(self.CL, key=len))
        k = max_CL_size - self.open_num
        if k > 0:
            f_benefit = {}
            for i in self.F - self.X:
                f_benefit[i] = - self.facility_costs[i]
                for j in self.P:
                    o = self.assign_client[j]
                    if self.cij[i][j] < self.cij[o][j]:
                        f_benefit[i] += self.cij[o][j] - self.cij[i][j]
            top_k = sorted(f_benefit.items(), key=lambda item: item[1], reverse=True)[:k]
            for i,value in top_k:
                self.X.add(i)



    def deal_CL(self):
        self.open_facility_for_CL()
        for CL in self.CL:
            F_list = list(self.X)
            cost_matrix = np.array([[self.cij[i][j] for j in CL] for i in F_list])
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for p, q in zip(row_ind, col_ind):
                i = F_list[p]
                j = CL[q]
                self.assign_client[j] = i


    def main(self):
        self.open_facility()
        CL_points = {p for item in self.CL for p in item}
        for j in (self.P - CL_points):
            best_i = min(self.X, key=lambda i: self.cij[i][j])
            self.assign_client[j] = best_i
        self.deal_CL()

def main(distance_matrix, Y, percent, i, facility_costs,file_name):
    start_time = time.time()
    FL = Greedy(distance_matrix, Y, facility_costs)
    FL.main()
    print("FL.open_f_count ", len(set(FL.assign_client)), len(FL.X))
    end_time = time.time()
    elapsed_time = end_time - start_time
    all_cost = evaluate.cost_out(FL.P, FL.cij, facility_costs, FL.assign_client,
                                 elapsed_time, f"output\\{file_name}\\{percent}%_Greedy_match_output_{i}.txt")
    return all_cost, elapsed_time


if __name__ == "__main__":
    percents = [2,4,6,8,10]
    facility_cost = []
    file_names = ['MP1','MP2','MP3','MP4','MP5']
    for file_name in file_names:
        with open(f'{file_name}_Greedy_match_result.txt', "a") as file:
            file.write(f"percent \t cost \t Time \n")
            for percent in percents:
                f_value_sum = 0
                elapsed_time_sum = 0
                for i in range(1, 101):
                    facility_costs, distance_matrix = data_deal_FL.read_file(0,file_name)
                    Y = data_deal_FL.read_CL(percent, i,file_name)
                    f_value, elapsed_time = main(distance_matrix, Y, percent, i, facility_costs,file_name)
                    f_value_sum += f_value
                    elapsed_time_sum += elapsed_time
                f_value_avg = f_value_sum / 100
                elapsed_time_avg = elapsed_time_sum / 100
                file.write(f"{percent}% \t {f_value_avg}  \t {elapsed_time_avg}\n")

