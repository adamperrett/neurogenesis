import csv
from ast import literal_eval

# f = open('wdbc.txt', 'r')
# breast_data = []
# for test in f:
#     breast_data.append(test)

breast_data = []
with open('wdbc.txt') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        breast_row = []
        for ele in row:
            if ele == 'M' or ele == 'B':
                breast_row.append(ele)
            else:
                breast_row.append(literal_eval(ele))
        breast_data.append(breast_row)


breast_labels = []
min_max = [[i, i] for i in breast_data[0][2:]]
for breast in breast_data:
    if breast[1] == 'B':
        breast_labels.append(0)
    else:
        breast_labels.append(1)
    for idx in range(len(min_max)):
        if breast[idx+2] < min_max[idx][0]:
            min_max[idx][0] = breast[idx+2]
        if breast[idx+2] > min_max[idx][1]:
            min_max[idx][1] = breast[idx+2]

norm_breast = []
for breast in breast_data:
    normed_breast = []
    for idx in range(len(min_max)):
        val_range = min_max[idx][1] - min_max[idx][0]
        normed_breast.append((breast[idx+2] - min_max[idx][0]) / val_range)
    norm_breast.append(normed_breast)

print("normalised breast")

from copy import deepcopy
training_set_breasts = deepcopy(norm_breast)
training_set_labels = deepcopy(breast_labels)
test_set_size = int(len(breast_data) * 0.3)
import random
test_set_indexes = random.sample([i for i in range(len(breast_labels))], test_set_size)
test_set_breasts = [norm_breast[i] for i in test_set_indexes]
test_set_labels = [breast_labels[i] for i in test_set_indexes]
test_set_indexes.sort(reverse=True)
for i in test_set_indexes:
    del training_set_breasts[i]
    del training_set_labels[i]