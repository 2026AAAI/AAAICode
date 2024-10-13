import copy
import time
import data_deal
import numpy as np
import math
import evaluate
class Facility_location():
    def __init__(self, dis,category,Y, f_cost, percent):
        self.cij = dis
        self.category = category
        self.client_num = len(category)
        self.CL = Y
        self.alpha_j = [0] * self.client_num
        self.target_num = len(np.unique(self.category))
        self.P = set(range(0, self.client_num))
        self.k = [-1] * self.client_num
        self.normal_c = None
        self.special_c = None
        self.assign_client = [-1] * self.client_num
        self.S = set(range(0, self.client_num))
        self.open_f = set()
        self.close_f = set(range(0, self.client_num))
        self.Yik = [[0] * len(self.CL) for _ in range(self.client_num)]
        self.gamma = [0] * self.client_num
        self.Rk = {}
        self.facility_cost = [f_cost] * self.client_num
        self.percent = percent
        self.is_step = 1
        self.delta = 1
        self.loop = 0
        self.stop_special = set()
        self.open_f_count = 0

    def init_data(self):
        Y_points = {p for item in self.CL for p in item }
        self.special_c = set(Y_points)
        self.normal_c = set(self.P - self.special_c)
        to_remove = set()
        for i in self.close_f:
            if self.facility_cost[i] == 0:
                to_remove.add(i)
        self.close_f -= to_remove
        self.open_f |= to_remove
        for i,item in enumerate(self.CL):
            self.Rk[i] = set()
            for key in item:
                self.k[key] = i

    def find_min_value(self, d):
        if not d:
            return None, float('inf')
        min_key = min(d, key=d.get)
        return min_key, d[min_key]


    def update_alpha_Yik(self):
        for j in self.S | self.special_c:
            self.alpha_j[j] += self.delta
        for j in self.special_c - self.S - self.stop_special:
            if self.gamma[j] == 0:
                continue
            o = self.assign_client[j]
            k = self.k[j]
            self.Yik[o][k] += self.delta

    def print_gamma(self):
        a = []
        for j in self.special_c:
            a.append(self.gamma[j])
        print("gamma ",a)

    def print_Yik(self):
        a = []
        for j in self.special_c:
            o = self.assign_client[j]
            k = self.k[j]
            a.append([j,k,o,self.Yik[o][k]])
        print("Yik ",a)


    def facility_is_open(self,i):
        fi = 0
        for j in self.S:
            fi += max(0, self.alpha_j[j]-self.cij[i][j])
        for j in self.P - self.S - self.special_c:
            o = self.assign_client[j]
            fi += max(0, self.cij[o][j] - self.cij[i][j])
        for j in self.special_c - self.S - self.stop_special:
            o = self.assign_client[j]
            k = self.k[j]
            fi += max(0, self.cij[o][j] + self.Yik[o][k] - self.cij[i][j])
        return fi > self.facility_cost[i]

    def deal_specila_facility(self,i):
        for j in (self.special_c - self.S):
            o = self.assign_client[j]
            if o == i or j in self.stop_special:
                continue
            k = self.k[j]
            this_o = [t for t in self.CL[k] if self.assign_client[t] == o]
            count_i = sum(1 for t in self.CL[k] if self.assign_client[t] == i)
            if len(this_o) == 1 and count_i == 0:
                if self.cij[o][j] > self.cij[i][j]:
                    self.assign_client[j] = i
                    self.Rk[k].add(i)
                    self.Yik[i][k] = self.cij[o][j] + self.Yik[o][k] - self.cij[i][j]
                    if this_o == 1:
                        self.Yik[o][k] = 0
                        self.Rk[k].discard(o)
            elif len(this_o) > 1 and count_i == 0:
                max_j = max(this_o, key =lambda t: self.Yik[o][k] + self.cij[o][t] - self.cij[i][t])
                if self.Yik[o][k] + self.cij[o][max_j] - self.cij[i][max_j] > 0:
                    self.assign_client[max_j] = i
                    self.Rk[k].add(i)
                    self.Yik[i][k] = self.cij[o][max_j] + self.Yik[o][k] - self.cij[i][max_j]
            if len(self.Rk[k]) == len(self.CL[k]):
                self.stop_special.update(self.CL[k])
                # if k == 0:
                #     print([self.assign_client[b] for b in self.CL[k]])
                #     print(self.Rk[k])
                #     print(self.CL[k])
                #     print(j,o,i)
                #     print(self.cij[0][301],self.cij[0][1345],self.Yik[0][0])
                #     print(self.cij[1460][301],self.cij[1460][1345],self.Yik[1460][0])
                #     print(self.cij[6][301],self.cij[6][1345],self.Yik[6][0])
                #     print("+++++++++++++++++++++++++++")

    def deal_speacil_contribution(self,i):
        for j in self.special_c - self.S:
            if j in self.stop_special:
                continue
            o = self.assign_client[j]
            k = self.k[j]
            this_o = sum(1 for t in self.CL[k] if self.assign_client[t] == o)
            if self.cij[o][j] + self.Yik[o][k] >= self.cij[i][j]:
                self.assign_client[j] = i
                self.Rk[k].add(i)
                self.Yik[i][k] = 0
                if this_o == 1:
                    self.Yik[o][k] = 0
                    self.Rk[k].discard(o)

    def is_break(self):
        self.loop += 1
        flag = (len(self.S) == 0)
        for i,item in enumerate(self.CL):
            if len(self.Rk[i]) != len(item):
                unique_clients = np.unique(self.assign_client)
                # print(self.Rk[i],"----",item,"-----",len(self.open_f),len(unique_clients[unique_clients != -1]))
                flag = False
            else:
                self.stop_special.update(item)
        return flag

    def cal_facility(self):
        while True:
            # print(len(self.S))
            self.update_alpha_Yik()
            # 未分配普通点-已经开放设施
            to_remove = set()
            for j in self.S - self.special_c:
                valid_i = [i for i in self.open_f if self.alpha_j[j] >= self.cij[i][j]]
                if not valid_i:
                    break
                min_i = min(valid_i, key=lambda i: self.cij[i][j])
                self.assign_client[j] = min_i
                to_remove.add(j)
            self.S -= to_remove

            # 遍历已经开放的设施
            for i in self.open_f:
                to_remove = set()
                # 未分配特殊点
                if self.S - self.normal_c:
                    for k,yk in enumerate(self.CL):
                        intersection = set(yk) & self.S
                        if not intersection:
                            continue
                        max_j = max(intersection, key =lambda j: self.alpha_j[j] - self.cij[i][j])
                        if self.alpha_j[max_j] >= self.cij[i][max_j] + self.Yik[i][k]:
                            to_remove.add(max_j)
                            self.gamma[max_j] = 1
                            self.Yik[i][k] = self.alpha_j[max_j] - self.cij[i][max_j]
                            self.assign_client[max_j] = i
                            self.Rk[k].add(i)
                self.S -= to_remove
                # 处理已分配特殊点
                self.deal_specila_facility(i)
            # 遍历未开放设施是否开放
            f_to_remove_set = set()
            for i in self.close_f:
                if self.facility_is_open(i):
                    f_to_remove_set.add(i)
                    to_remove = set()
                    # 未开放的普通点
                    for j in self.S - self.special_c:
                        if self.alpha_j[j] >= self.cij[i][j]:
                            self.assign_client[j] = i
                            to_remove.add(j)
                    for j in self.S & self.special_c:
                        if self.alpha_j[j] >= self.cij[i][j]:
                            self.assign_client[j] = i
                            k = self.k[j]
                            self.Rk[k].add(i)
                            self.gamma[j] = 1
                            to_remove.add(j)
                    self.S -= to_remove
                    # 已经分配的普通点
                    for j in (self.P - self.S - self.special_c):
                        o = self.assign_client[j]
                        if self.cij[o][j] > self.cij[i][j]:
                            self.assign_client[j] = i
                    # 处理已经分配的特殊点
                    self.deal_speacil_contribution(i)

            self.close_f -= f_to_remove_set
            self.open_f |= f_to_remove_set

            if self.is_break():
                break
        return

    def main(self):
        self.init_data()
        self.cal_facility()
        self.open_f_count = len(np.unique(self.assign_client))

def main(dis,category,Y,percent,i):
    start_time = time.time()
    lambda_upper = np.sum(dis)/4000
    lambda_lower = 0
    while True:
        lambda_middle = (lambda_upper + lambda_lower) / 2
        print("lambda_middle ",lambda_middle)
        FL = Facility_location(dis,category,Y, lambda_middle,percent)
        FL.main()
        print("FL.open_f_count ",FL.open_f_count)
        if FL.open_f_count < FL.target_num:
            lambda_upper = lambda_middle
        elif FL.open_f_count > FL.target_num:
            lambda_lower = lambda_middle
        else:
            end_time = time.time()
            elapsed_time = end_time - start_time
            f_value,purity,RI,nmi = evaluate.output(FL.P,FL.cij,FL.category,FL.assign_client,
                                                    elapsed_time,f"{percent}%_faciliti_output_{i}.txt")
            break
    return f_value,purity,RI,nmi,elapsed_time

if __name__ == "__main__":
    percents = [2,4,6,8,10]
    with open("facility_result.txt", "a") as file:
        file.write(f"percent \t cost \t purity\t RI \t NMI \t Time \n ")
        for percent in percents:
            f_value_sum = 0
            purity_sum = 0
            RI_sum = 0
            nmi_sum = 0
            elapsed_time_sum = 0
            for i in range(1,2):
                dis,category= data_deal.read_obesity_data(percent,i,0)
                Y = data_deal.read_CL(percent,i)
                f_value,purity,RI,nmi,elapsed_time = main(dis,category,Y,percent,i)
                f_value_sum += f_value
                purity_sum += purity
                RI_sum += RI
                nmi_sum += nmi
                elapsed_time_sum += elapsed_time
            f_value_avg = f_value_sum / 1
            purity_avg = purity_sum / 1
            RI_avg = RI_sum / 1
            nmi_avg = nmi_sum / 1
            elapsed_time_avg = elapsed_time_sum /1
            file.write(f"{percent}% \t {f_value_avg} \t {purity_avg} \t {RI_avg} \t {nmi_avg} \t {elapsed_time_avg}\n")
