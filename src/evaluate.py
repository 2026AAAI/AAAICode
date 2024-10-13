from collections import Counter
import copy
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import rand_score
import numpy as np
from sklearn.metrics import confusion_matrix
def cal_f_value(P,assign_client,cij):
    f = 0
    for j in P:
        o = assign_client[j]
        f += cij[o][j]
    return f

def split_group(category):
    group = {}
    for i, value in enumerate(category):
        if value not in group:
            group[value] = []
        group[value].append(i)
    list_group = []
    for value in group.values():
        list_group.append(value)
    return list_group

def cal_purity(category,assign_client):
    # S = split_group(category)
    # L = split_group(assign_client)
    # n = sum(len(cluster) for cluster in S)
    # purity_score = 0
    # for cluster in S:
    #     # 对于每个聚类，找到与标签 Lj 的最大交集
    #     max_intersection = max(len(set(cluster) & set(label)) for label in L)
    #     purity_score += max_intersection
    # return purity_score / n
    contingency_matrix = confusion_matrix(category, assign_client)
    total_samples = np.sum(contingency_matrix)
    max_in_clusters = np.sum(np.amax(contingency_matrix, axis=0))
    purity = max_in_clusters / total_samples
    return purity

def cal_RI(category,assign_client):
    # similar = 0
    # num = len(category)
    # for i in range(num):
    #     for j in range(i+1,num):
    #         if category[i]==category[j] and assign_client[i] == assign_client[j]:
    #             similar += 1
    #         if category[i] != category[j] and assign_client[i] != assign_client[j]:
    #             similar += 1
    # RI = similar / (num * (num - 1)/2)
    RI = rand_score(category, assign_client)
    return RI


def cal_NMI(category,assign_client):
    nmi = normalized_mutual_info_score(category, assign_client)
    return nmi


def output(P,cij,category,assign_client,elapsed_time,output_file):
    purity = cal_purity(category,assign_client)
    f_value = cal_f_value(P,assign_client,cij)
    RI = cal_RI(category,assign_client)
    nmi = cal_NMI(category,assign_client)
    with open(output_file, 'w') as f:
        f.write(f"cost: {f_value} \t purity: {purity} \t RI: {RI} \t NMI: {nmi}\n")
        f.write(f"time: {elapsed_time} s\n")
        for j in P:
            o = assign_client[j]
            f.write(f"{j} {o}\n")
    return f_value,purity,RI,nmi
