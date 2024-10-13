import copy
import time
import data_deal
import numpy as np
import math
import evaluate
import random
class Facility_location():
    def __init__(self, dis,category, f_cost):
        self.cij = dis
        self.category = category
        self.client_num = len(category)
        self.alpha_j = [0] * self.client_num
        self.target_num = len(np.unique(self.category))
        self.P = set(range(0, self.client_num))
        self.assign_client = [-1] * self.client_num
        self.S = set(range(0, self.client_num))
        self.open_f = set()
        self.close_f = set(range(0, self.client_num))
        self.facility_cost = [f_cost] * self.client_num
        self.open_f_count = 0
        self.step = 0.1
        self.is_step = 0

    def init_data(self):
        to_remove = set()
        for i in self.close_f:
            if self.facility_cost[i] == 0:
                to_remove.add(i)
        self.close_f -= to_remove
        self.open_f |= to_remove

    def find_min_value(self, d):
        if not d:
            return None, float('inf')
        min_key = min(d, key=d.get)
        return min_key, d[min_key]

    # 直接计算满足条件时，alpha要增加多少
    def cal_increase(self):
        fi = copy.deepcopy(self.facility_cost)
        # 计算让一个设施从未开放变为开放，需要的最小alpha值
        f_close_alpha = {}
        for i in self.close_f:
            count = 0
            for j in self.S:
                count += 1
                fi[i] += self.cij[i][j]
            for j in self.P - self.S:
                o = self.assign_client[j]
                fi[i] -= max(0, self.cij[o][j] - self.cij[i][j])

            # 计算设施i想要开放需要连接用户的alpha值
            f_close_alpha[i] = fi[i] / count
        min_close_key, min_close_value = self.find_min_value(f_close_alpha)

        delta = min_close_value
        for j in self.S:
            self.alpha_j[j] = self.alpha_j[j] + max(1, delta - self.alpha_j[j]) + 1e-6

        # print(delta,min_close_value,min_close_key)
        # print(f_close_alpha[min_close_key] * 299,fi[min_close_key])

    def update_alpha(self):
        if self.is_step:
            for j in self.S:
                self.alpha_j[j] += self.step
        else:
            self.cal_increase()

    def facility_is_open(self,i):
        fi = 0
        for j in self.P:
            if self.assign_client[j] != -1:
                o = self.assign_client[j]
                fi += max(0, self.cij[o][j] - self.cij[i][j])
            else:
                fi += max(0, self.alpha_j[j]-self.cij[i][j])
        # if i == 106:
        #     print("106 ",fi,self.facility_cost[i])
        return fi + 1e-6> self.facility_cost[i]

    def cal_facility(self):
        while len(self.S) > 0:
            # print(len(self.S))
            self.update_alpha()
            to_remove = set()

            for j in self.S:
                valid_i = [i for i in self.open_f if self.alpha_j[j] >= self.cij[i][j]]
                if not valid_i:
                    break
                min_i = min(valid_i, key=lambda i: self.cij[i][j])
                self.assign_client[j] = min_i
                to_remove.add(j)
            self.S -= to_remove

            # 遍历未开放设施是否开放
            f_to_remove_set = set()
            for i in self.close_f:
                if self.facility_is_open(i):
                    f_to_remove_set.add(i)
                    to_remove = set()
                    for j in self.S:
                        if self.alpha_j[j] > self.cij[i][j]:
                            self.assign_client[j] = i
                            to_remove.add(j)
                    self.S -= to_remove
                    for j in (self.P - self.S):
                        o = self.assign_client[j]
                        if self.cij[o][j] > self.cij[i][j]:
                            self.assign_client[j] = i
            self.close_f -= f_to_remove_set
            self.open_f |= f_to_remove_set
        return


    def main(self):
        self.init_data()
        self.cal_facility()
        self.open_f_count = len(self.open_f)

def main(dis,category):
    start_time = time.time()
    lambda_upper = np.sum(dis) /10000
    lambda_lower = 0
    while True:
        lambda_middle = (lambda_upper + lambda_lower) / 2
        print("lambda_middle ",lambda_middle)
        FL = Facility_location(dis,category, lambda_middle)
        FL.main()
        print("FL.open_f_count ",FL.open_f_count,FL.target_num)
        if FL.open_f_count < FL.target_num:
            lambda_upper = lambda_middle
        elif FL.open_f_count > FL.target_num:
            lambda_lower = lambda_middle
        else:
            end_time = time.time()
            elapsed_time = end_time - start_time
            # evaluate.output(FL.P,FL.cij,FL.category,FL.assign_client,
            #                 elapsed_time,f"{percent}%_compare_output_{i}.txt")
            break
    return FL,elapsed_time

def cal_CL(FL,f_time,Y,percent,i):
    start_time = time.time()
    for CL in Y:
        item_categories = set([FL.assign_client[i] for i in CL])
        if len(set(item_categories)) == len(CL):
            continue
        all_categories = set(FL.assign_client)
        for j in CL:
            e = random.choice(list(all_categories))
            FL.assign_client[j] = e
            all_categories.discard(e)
    end_time = time.time()
    elapsed_time = end_time - start_time
    f_value,purity,RI,nmi = evaluate.output(FL.P,FL.cij,FL.category,FL.assign_client,
                                            elapsed_time + f_time,f"{percent}%_compare_output_{i}.txt")

    return f_value,purity,RI,nmi,elapsed_time


if __name__ == "__main__":
    dis,category= data_deal.read_obesity_data(0,0,0)
    a_FL,cal_elapsed_time = main(dis,category)
    evaluate.output(a_FL.P,a_FL.cij,a_FL.category,a_FL.assign_client,
                    cal_elapsed_time,f"compare_output.txt")
    with open("compare_result.txt", "a") as file:
        file.write(f"percent \t cost \t purity\t RI \t NMI \t Time \n")
        for percent in [2,4,6,8,10]:
            FL = copy.deepcopy(a_FL)
            f_value_sum = 0
            purity_sum = 0
            RI_sum = 0
            nmi_sum = 0
            elapsed_time_sum = 0
            for i in range(1,11):
                dis,category= data_deal.read_obesity_data(percent,i,0)
                FL = copy.deepcopy(a_FL)
                # FL,cal_elapsed_time = main(dis,category)
                Y = data_deal.read_CL(percent,i)
                f_value,purity,RI,nmi,elapsed_time = cal_CL(FL,cal_elapsed_time,Y,percent,i)
                f_value_sum += f_value
                purity_sum += purity
                RI_sum += RI
                nmi_sum += nmi
                elapsed_time_sum += elapsed_time+cal_elapsed_time
            f_value_avg = f_value_sum / 10
            purity_avg = purity_sum / 10
            RI_avg = RI_sum / 10
            nmi_avg = nmi_sum / 10
            elapsed_time_avg = elapsed_time_sum /10
            file.write(f"{percent}% \t {f_value_avg} \t {purity_avg} \t {RI_avg} \t {nmi_avg} \t {elapsed_time_avg}\n")