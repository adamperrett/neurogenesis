import numpy as np
import csv
from copy import deepcopy

print('reading data')
mpg_features = []
mpg_values = []
with open('../datasets/mpg regression dataset.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in reader:
        mpg_values.append(float(row[1]))
        parsed_feat = []
        for val in row[2:]:
            if val == '?':
                parsed_feat.append('?')
            else:
                parsed_feat.append(float(val))
        mpg_features.append(parsed_feat)
        # print(', '.join(row))

print('determining value range')
# determine boundary for nornalisation
min_features = deepcopy(mpg_features[-1])
max_features = deepcopy(mpg_features[-1])
for car in mpg_features:
    for val in range(len(mpg_features[0])):
        if car[val] == '?':
            continue
        if car[val] < min_features[val]:
            min_features[val] = car[val]
        if car[val] > max_features[val]:
            max_features[val] = car[val]

min_mpg = 1000
max_mpg = 0
for car in mpg_values:
    if car < min_mpg:
        min_mpg = car
    if car > max_mpg:
        max_mpg = car

print('normalising')
# normalise
features_to_normalise = [0, 1, 2, 3, 4, 5]
norm_features = []
for car in mpg_features:
    norm_feat = []
    for feat in range(len(mpg_features[0])):
        if car[feat] == '?':
            norm_feat.append(1000)
        else:
            if feat in features_to_normalise:
                new_val = (car[feat] - min_features[feat]) / (max_features[feat] - min_features[feat])
                norm_feat.append(new_val)
            else:
                norm_feat.append(car[feat])
    norm_features.append(norm_feat)

norm_mpg = []
for car in mpg_values:
    norm_mpg.append(car)#(car - min_mpg) / (max_mpg - min_mpg))

print('done')

