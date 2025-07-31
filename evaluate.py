from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import rand_score
import numpy as np
from sklearn.metrics import confusion_matrix


def cal_f_value(P, assign_client, cij):
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


def cal_purity(category, assign_client):
    contingency_matrix = confusion_matrix(category, assign_client)
    total_samples = np.sum(contingency_matrix)
    max_in_clusters = np.sum(np.amax(contingency_matrix, axis=0))
    purity = max_in_clusters / total_samples
    return purity


def cal_RI(category, assign_client):
    RI = rand_score(category, assign_client)
    return RI


def cal_NMI(category, assign_client):
    nmi = normalized_mutual_info_score(category, assign_client)
    return nmi


def output(P, cij, category, assign_client, elapsed_time, output_file):
    purity = cal_purity(category, assign_client)
    f_value = cal_f_value(P, assign_client, cij)
    RI = cal_RI(category, assign_client)
    nmi = cal_NMI(category, assign_client)
    with open(output_file, 'w') as f:
        f.write(f"cost: {f_value} \t purity: {purity} \t RI: {RI} \t NMI: {nmi}\n")
        f.write(f"time: {elapsed_time} s\n")
        for j in P:
            o = assign_client[j]
            f.write(f"{j} {o}\n")
    return f_value, purity, RI, nmi


def cost_out(P, cij, fi, assign_client, elapsed_time, output_file):
    all_cost = 0
    for i in set(assign_client):
        all_cost += fi[i]

    for j in P:
        i = assign_client[j]
        all_cost += cij[i][j]
    with open(output_file, 'w') as f:
        f.write(f"cost: {all_cost} \n")
        f.write(f"time: {elapsed_time} s\n")
        for j in P:
            o = assign_client[j]
            f.write(f"{j} {o}\n")
    return all_cost
