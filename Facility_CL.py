import copy
import time
import data_deal_FL
import numpy as np
import evaluate
import networkx as nx
from scipy.optimize import linear_sum_assignment

class Facility():
    def __init__(self, distance_matrix, Y, facility_costs):
        self.cij = distance_matrix
        self.client_num = len(distance_matrix[0])
        self.facility_num = len(facility_costs)
        self.CL = Y
        # self.F: all facilities
        self.F = set(range(0, self.facility_num))
        self.X = set()
        # self.P: all clients; self.S: unassigned clients
        self.P = set(range(0, self.client_num))
        self.S = set(range(0, self.client_num))
        self.special = set(p for item in self.CL for p in item)
        self.normal = set(self.P - self.special)
        self.CL_facility = [set() for CL in self.CL]
        self.assign_client = [-1] * self.client_num
        self.facility_costs = copy.deepcopy(facility_costs)
        self.open_num = 0
        self.special_to_CL = {}

    def init_data(self):
        for row_index, CL in enumerate(self.CL):
            for value in CL:
                self.special_to_CL[value] = row_index

    def cal_normal_benefit(self, i):
        # Calculate the benefit of reassigning already assigned normal clients to facility i, i.e., distance reduction
        benefit = 0
        for j in (self.P - self.S):
            current_cost = min(self.cij[k][j] for k in self.X)
            if current_cost > self.cij[i][j]:
                benefit += (current_cost - self.cij[i][j])
        return benefit

    def cal_special_cost(self, i, facility_flow_info, facility_CL_point_cost):
        for idx, CL in enumerate(self.CL):
            # Skip if this CL group already has clients in facility i or the whole group is already assigned
            if i in self.CL_facility[idx] or len(self.CL_facility[idx]) == len(CL):
                continue
            # For each CL group, perform a min-cost flow calculation
            G = nx.DiGraph()
            G.add_edge('s', 'y', capacity=1, weight=0)
            for j in CL:
                if self.assign_client[j] == -1:
                    G.add_edge('y', j, capacity=1, weight=0)
                else:
                    G.add_edge('y', j, capacity=0, weight=0)
                # Add edge from client to current facility
                G.add_edge(j, self.client_num + i, capacity=1, weight=int(self.cij[i][j]))
                # If the client is already assigned to another open facility:
                # add negative edge weight and 0 capacity from y->client
                for o in self.X:
                    if self.assign_client[j] == o:
                        G.add_edge(self.client_num + o, j, capacity=0, weight=int(-self.cij[i][o]))
                    else:
                        G.add_edge(j, self.client_num + o, capacity=1, weight=int(self.cij[i][o]))
            # Compute flow path and get min cost: cost of assigning the CL group to facility i
            flow_dict = nx.max_flow_min_cost(G, 's', self.client_num + i)
            min_cost = nx.cost_of_flow(G, flow_dict)
            # Store flow and point assigned to facility i (may be exchanged from another facility)
            point, tmp_flow = self.deal_flow(flow_dict, G, self.client_num + i)
            facility_flow_info[i][point] = tmp_flow
            if point == -1:
                print("point bug")
                exit(0)
            # Store cost of assigning point in CL group to facility i
            facility_CL_point_cost[i][point] = min_cost

    def deal_flow(self, flow_dict, G, i):
        tmp_flow = []
        for u in flow_dict:
            for v, flow in flow_dict[u].items():
                if flow <= 0: continue
                cost = G[u][v]['weight']
                if cost < 0 or not str(u).isdigit() or not str(v).isdigit(): continue
                tmp_flow.append((u, v))
        for (u, v) in tmp_flow:
            if v == i:
                return u, tmp_flow
        return -1, tmp_flow

    def cal_sorted_cost_point(self, i, facility_CL_point_cost):
        unsort_point_cost = {}
        # Cost of unassigned normal clients is cij
        for j in self.S - self.special:
            unsort_point_cost[j] = self.cij[i][j]
        # Cost of special clients in CL is stored in facility_CL_point_cost[i][point]
        for point, cost in facility_CL_point_cost[i].items():
            unsort_point_cost[point] = cost

        # Sort in ascending order
        sorted_point_cost = dict(
            sorted(unsort_point_cost.items(), key=lambda item: item[1])
        )
        return sorted_point_cost

    def open_facility(self):
        while self.S:
            best_ratio = float('inf')
            best_i = None
            best_Y = set()

            facility_flow_info = {}
            facility_CL_point_cost = {}
            for i in self.F:
                facility_flow_info[i] = {}
                facility_CL_point_cost[i] = {}
                normal_benefit = self.cal_normal_benefit(i)
                # Calculate facility_flow_info and facility_CL_point_cost
                self.cal_special_cost(i, facility_flow_info, facility_CL_point_cost)
                # Sort unassigned normal and special clients in CL, compute prefix cost
                sorted_point_cost = self.cal_sorted_cost_point(i, facility_CL_point_cost)
                # If no point can be assigned to this facility, skip it
                # This only happens when remaining unassigned special points can't be assigned due to conflicts
                if not sorted_point_cost: continue
                prefix_cost = 0
                prev_ratio = float('inf')
                for idx, (j, cost) in enumerate(sorted_point_cost.items(), 1):
                    prefix_cost += cost
                    ratio = (self.facility_costs[i] - normal_benefit + prefix_cost) / idx
                    if ratio > prev_ratio:
                        break
                    else:
                        if ratio < best_ratio:
                            best_ratio = ratio
                            best_i = i
                            best_Y = set(list(sorted_point_cost)[:idx])
                    prev_ratio = ratio
            # Assign clients in Y to facility best_i
            for j in best_Y:
                self.assign_client[j] = best_i
            # Handle reassignments for normal clients already assigned to other facilities
            for j in self.normal - self.S:
                o = self.assign_client[j]
                if self.cij[best_i][j] < self.cij[o][j]:
                    self.assign_client[j] = best_i

            # Handle special client assignment (may involve exchange)
            for j in best_Y:
                if j not in self.special: continue
                if j not in facility_flow_info[best_i]:
                    print("facility_flow_info[best_i] bug")
                    exit(0)
                for (u, v) in facility_flow_info[best_i][j]:
                    self.assign_client[u] = v - self.client_num

            # Update state
            self.facility_costs[best_i] = 0
            self.X.add(best_i)
            self.S -= best_Y

            self.CL_facility = [set() for CL in self.CL]
            # Update CL-to-facility mapping due to reassignment
            for j in self.special - self.S:
                CL_index = self.special_to_CL[j]
                self.CL_facility[CL_index].add(self.assign_client[j])

    def special_matching(self):
        for j in self.normal:
            best_i = min(self.X, key=lambda i: self.cij[i][j])
            self.assign_client[j] = best_i
        for CL in self.CL:
            F_list = list(self.X)
            cost_matrix = np.array([[self.cij[i][j] for j in CL] for i in F_list])
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            # Output matching results
            for p, q in zip(row_ind, col_ind):
                i = F_list[p]
                j = CL[q]
                self.assign_client[j] = i

    def main(self):
        self.init_data()
        self.open_facility()
        self.special_matching()


def main(distance_matrix, Y, percent, i, facility_costs, file_name):
    start_time = time.time()
    FL = Facility(distance_matrix, Y, facility_costs)
    FL.main()
    print("FL.open_f_count ", len(set(FL.assign_client)), len(FL.X))
    end_time = time.time()
    elapsed_time = end_time - start_time
    all_cost = evaluate.cost_out(FL.P, FL.cij, facility_costs, FL.assign_client,
                                 elapsed_time, f"output\\{file_name}\\{percent}%_facility_output_{i}.txt")
    return all_cost, elapsed_time


if __name__ == "__main__":
    percents = [2, 4, 6, 8, 10]
    facility_cost = []
    file_names = ['MP1', 'MP2', 'MP3', 'MP4', 'MP5']
    for file_name in file_names:
        with open(f'{file_name}_facility_result.txt', "a") as file:
            file.write(f"percent \t cost \t Time \n")
            for percent in percents:
                f_value_sum = 0
                elapsed_time_sum = 0
                for i in range(1, 101):
                    facility_costs, distance_matrix = data_deal_FL.read_file(0, file_name)
                    Y = data_deal_FL.read_CL(percent, i, file_name)
                    f_value, elapsed_time = main(distance_matrix, Y, percent, i, facility_costs, file_name)
                    f_value_sum += f_value
                    elapsed_time_sum += elapsed_time
                f_value_avg = f_value_sum / 100
                elapsed_time_avg = elapsed_time_sum / 100
                file.write(f"{percent}% \t {f_value_avg}  \t {elapsed_time_avg}\n")

