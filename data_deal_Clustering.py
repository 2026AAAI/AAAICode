import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import random
import numpy as np
from sklearn.preprocessing import LabelEncoder
from collections import Counter


def manhattan_distance_matrix(data):
    n = data.shape[0]
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist_matrix[i, j] = np.sum(np.abs(data[i] - data[j]))
            dist_matrix[j, i] = dist_matrix[i, j]  # The distance matrix is symmetric
    return dist_matrix


def heart_data():
    file_path = 'data\\original_data\\heart_failure_clinical_records_dataset.csv'
    data = pd.read_csv(file_path)
    features = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction',
                'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 'sex',
                'smoking', 'time']
    scaler = MinMaxScaler()
    data[features] = scaler.fit_transform(data[features])
    normalized_features = data[features].values
    manhattan_distances = manhattan_distance_matrix(normalized_features)
    distance_df = pd.DataFrame(manhattan_distances)
    distance_df['DEATH_EVENT'] = data['DEATH_EVENT'].values
    output_path = 'data\\normalized_heart_failure_data.csv'
    data.to_csv(output_path, index=False, encoding='utf-8')
    output_distance_path = 'data\\md_heart_failure_data.csv'
    distance_df.to_csv(output_distance_path, index=False, encoding='utf-8')


def wholesale_data():
    file_path = 'data\\original_data\\Wholesale_customers_data.csv'
    data = pd.read_csv(file_path)
    features = ['Channel', 'Fresh', 'Milk', 'Grocery', 'Frozen',
                'Detergents_Paper', 'Delicassen']
    scaler = MinMaxScaler()
    data[features] = scaler.fit_transform(data[features])
    normalized_features = data[features].values
    manhattan_distances = manhattan_distance_matrix(normalized_features)
    distance_df = pd.DataFrame(manhattan_distances)
    distance_df['Region'] = data['Region'].values
    output_path = 'data\\normalized_wholesale_customers_data.csv'
    data.to_csv(output_path, index=False, encoding='utf-8')
    output_distance_path = 'data\\md_wholesale_customers_data.csv'
    distance_df.to_csv(output_distance_path, index=False, encoding='utf-8')


def ObesityDataSet_change():
    file_path = 'data\\original_data\\ObesityDataSet_raw_and_data_sinthetic.csv'
    data = pd.read_csv(file_path)
    le = LabelEncoder()
    categorical_columns = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC',
                           'SMOKE', 'SCC', 'CALC', 'MTRANS', 'NObeyesdad']

    for col in categorical_columns:
        data[col] = le.fit_transform(data[col])
    output_file_path = 'data\\original_data\\processed_Obesity_dataset.csv'
    data.to_csv(output_file_path, index=False, encoding='utf-8')


def obesity_data():
    file_path = 'data\\original_data\\processed_Obesity_dataset.csv'
    data = pd.read_csv(file_path)

    features = ['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight',
                'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF',
                'TUE', 'CALC', 'MTRANS']
    scaler = MinMaxScaler()
    data[features] = scaler.fit_transform(data[features])
    normalized_features = data[features].values
    manhattan_distances = manhattan_distance_matrix(normalized_features)
    distance_df = pd.DataFrame(manhattan_distances)
    distance_df['NObeyesdad'] = data['NObeyesdad'].values
    output_path = 'data\\normalized_obesity_data.csv'
    data.to_csv(output_path, index=False, encoding='utf-8')
    output_distance_path = 'data\\md_obesity_data.csv'
    distance_df.to_csv(output_distance_path, index=False, encoding='utf-8')


def wine_data():
    file_path = 'data\\original_data\\wine.csv'
    data = pd.read_csv(file_path)
    column_names = data.columns.tolist()
    features = [
        "Alcohol", "Malicacid", "Ash", "Alcalinity_of_ash", "Magnesium",
        "Total_phenols", "Flavanoids", "Nonflavanoid_phenols", "Proanthocyanins",
        "Color_intensity", "Hue", "OD280_OD315_of_diluted_wines", "Proline"
    ]
    scaler = MinMaxScaler()
    data[features] = scaler.fit_transform(data[features])
    normalized_features = data[features].values
    manhattan_distances = manhattan_distance_matrix(normalized_features)
    distance_df = pd.DataFrame(manhattan_distances)
    distance_df['class'] = data['class'].values
    output_path = 'data\\normalized_wine_data.csv'
    data.to_csv(output_path, index=False, encoding='utf-8')
    output_distance_path = 'data\\md_wine_data.csv'
    distance_df.to_csv(output_distance_path, index=False, encoding='utf-8')


def student_data():
    file_path = 'data\\original_data\\student.csv'
    data = pd.read_csv(file_path)
    features = [
        "Marital_status", "Application_mode", "Application_order", "Course", "Attendance_type",
        "Previous_qualification", "Qualification_grade", "Nationality", "Mother's_qualification",
        "Father's_qualification", "Mother's_occupation", "Father's_occupation", "Admission_grade",
        "Displaced", "Special_needs", "Debtor", "Tuition_up_to_date", "Gender", "Scholarship_holder",
        "Age_at_enrollment", "International", "1st_sem_credited", "1st_sem_1", "1st_sem_evaluations",
        "1st_sem_approved", "1st_sem_grade", "1st_sem_without_evaluations", "2nd_sem_credited",
        "2nd_sem_1", "2nd_sem_evaluations", "2nd_sem_approved", "2nd_sem_grade",
        "2nd_sem_without_evaluations", "Unemployment_rate", "Inflation_rate", "GDP"
    ]
    scaler = MinMaxScaler()
    data[features] = scaler.fit_transform(data[features])
    normalized_features = data[features].values
    manhattan_distances = manhattan_distance_matrix(normalized_features)
    distance_df = pd.DataFrame(manhattan_distances)
    distance_df['Target'] = data['Target'].values
    output_path = 'data\\normalized_student_data.csv'
    data.to_csv(output_path, index=False, encoding='utf-8')
    output_distance_path = 'data\\md_student_data.csv'
    distance_df.to_csv(output_distance_path, index=False, encoding='utf-8')


def read_heart_failure(percent, i, gengrate_Y=0):
    file_path = 'data\\manhattan_data\\md_heart_failure_data.csv'
    data = pd.read_csv(file_path)
    distance_data = data.iloc[:, :-1].values
    category_data = data['DEATH_EVENT'].values
    dis = np.array(distance_data)
    category = np.array(category_data)
    if gengrate_Y:
        output_file = f'{percent}%CL_{i}.txt'
        generate_CL(output_file, category, percent)
    # output_file = 'weight_heart_failure.txt'
    # generate_weight(dis,category,output_file)
    return dis, category


def read_wholesale_customers(percent, i, gengrate_Y=0):
    file_path = 'data\\manhattan_data\\md_wholesale_customers_data.csv'
    data = pd.read_csv(file_path)
    distance_data = data.iloc[:, :-1].values
    category_data = data['Region'].values
    dis = np.array(distance_data)
    category = np.array(category_data)
    if gengrate_Y:
        output_file = f'{percent}%CL_{i}.txt'
        generate_CL(output_file, category, percent)
    return dis, category


def read_obesity_data(percent, i, gengrate_Y=0):
    file_path = 'data\\manhattan_data\\md_obesity_data.csv'
    data = pd.read_csv(file_path)
    distance_data = data.iloc[:, :-1].values
    category_data = data['NObeyesdad'].values
    dis = np.array(distance_data)
    category = np.array(category_data)
    if gengrate_Y:
        output_file = f'{percent}%CL_{i}.txt'
        generate_CL(output_file, category, percent)
    return dis, category


def read_wine_data(percent, i, gengrate_Y=0):
    file_path = 'data\\manhattan_data\\md_wine_data.csv'
    data = pd.read_csv(file_path)
    distance_data = data.iloc[:, :-1].values
    category_data = data['class'].values
    dis = np.array(distance_data)
    category = np.array(category_data)
    if gengrate_Y:
        output_file = f'{percent}%CL_{i}.txt'
        generate_CL(output_file, category, percent)
    return dis, category


def read_student_data(percent, i, gengrate_Y=0):
    file_path = 'data\\manhattan_data\\md_student_data.csv'
    data = pd.read_csv(file_path)
    distance_data = data.iloc[:, :-1].values
    category_data = data['Target'].values
    dis = np.array(distance_data)
    category = np.array(category_data)
    if gengrate_Y:
        output_file = f'{percent}%CL_{i}.txt'
        generate_CL(output_file, category, percent)
    return dis, category


def generate_CL(output_file, category, percent):
    k = len(category) * percent / 100
    unique_categories = np.unique(category)
    Y = []
    with open(output_file, 'w') as file:
        count = 0
        selected_points = set()
        flag = False
        while len(selected_points) < k:
            y = []
            for cat in unique_categories:
                category_indices = np.where(category == cat)[0]
                available_indices = [idx for idx in category_indices if idx not in selected_points]  # 过滤掉已经选择的点
                if len(available_indices) > 0:
                    if not flag or random.random() < 0.5:
                        random_point = random.choice(available_indices)
                        y.append(random_point)
                        selected_points.add(random_point)
            flag = True
            if (len(y) >= 2):
                file.write(str(y) + '\n')
                count += 1
                Y.append(y)
            else:
                for t in y:
                    selected_points.remove(t)


def read_CL(percent, i):
    file = f'{percent}%CL_{i}.txt'
    Y = []
    with open(file, 'r') as f:
        for line in f:
            pair = line.strip().strip('[]').split(',')
            Y.append([int(item) for item in pair])
    return Y


if __name__ == "__main__":
    for percent in [2, 4, 6, 8, 10]:
        for i in range(1, 101):
            read_obesity_data(percent, i, 1)
    # heart_data()
    # wholesale_data()
    # obesity_data()
    # student_data()

