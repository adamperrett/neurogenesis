import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

def test_calc_times(short_length, long_length):
    long_dict = {}
    for i in range(long_length):
        long_dict['{}'.format(i)] = np.random.random()
    short_dict = {}
    for i in range(short_length):
        # short_dict['{}'.format(i+(long_length/2))] = 100 - i
        short_dict['{}'.format(i)] = np.random.random()

    repeat = 10000

    time_original = []
    time_matrix = []
    time_comp = []

    # print("Doing original")
    for i in range(repeat):
        t0 = time.time()

        all_values = 0.
        for ele in short_dict:
            if ele in long_dict:
                all_values += 1 - np.abs((short_dict[ele] - long_dict[ele]) / 0.6)

        t1 = time.time()
        time_original.append(t1 - t0)
        # print(all_values)

    # print("Doing matrix")
    for i in range(repeat):
        t0 = time.time()

        # all_values = 0.
        short_values = np.array(list(short_dict.values()))
        long_values = {k: long_dict[k] for k in short_dict}
        long_values = np.array(list(long_values.values()))
        all_values = np.sum(1 - np.abs(np.subtract(short_values, long_values) / 0.6))

        t1 = time.time()
        time_matrix.append(t1 - t0)
        # print(all_values)

    # print("Doing list comp")
    short_values = np.array(list(short_dict.values()))
    long_values = {k: long_dict[k] for k in short_dict}
    long_values = np.array(list(long_values.values()))
    for i in range(repeat):
        # all_values = np.sum([1 - np.abs((short_dict[ele] - long_dict[ele]) / 0.6) for ele in short_dict])

        t0 = time.time()

        # all_values = 0.
        all_values = np.sum(1 - np.abs(np.subtract(short_values, long_values) / 0.6))

        t1 = time.time()
        time_comp.append(t1 - t0)
        # print(all_values)

    # total_time = np.average(time_original) + np.average(time_matrix) + np.average(time_comp)
    total_time = np.min([np.average(time_original), np.average(time_matrix), np.average(time_comp)])
    print("\nwith short length", short_length, "and long length", long_length)
    print(np.average(time_original) / total_time, "time for original calc = ", np.average(time_original), "with sdtev", np.std(time_original))
    print(np.average(time_matrix) / total_time, "time for numpy calc = ", np.average(time_matrix), "with sdtev", np.std(time_matrix))
    print(np.average(time_comp) / total_time, "time for list compression = ", np.average(time_comp), "with sdtev", np.std(time_comp))

    return np.average(time_original) / total_time, np.average(time_original), \
           np.average(time_matrix) / total_time, np.average(time_matrix), \
           np.average(time_comp) / total_time, np.average(time_comp)

lengths = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

all_data = []
original_share = [[0 for i in range(len(lengths))] for j in range(len(lengths))]
matrix_share = [[0 for i in range(len(lengths))] for j in range(len(lengths))]
comp_share = [[0 for i in range(len(lengths))] for j in range(len(lengths))]
for x, sl in enumerate(lengths):
    for y, ll in enumerate(lengths):
        if sl <= ll:
            result = test_calc_times(sl, ll)
            all_data.append(result)
            original_share[x][y] = result[0]
            matrix_share[x][y] = result[2]
            comp_share[x][y] = result[4]


fig, axs = plt.subplots(3, 1)

df_original = pd.DataFrame(original_share, range(len(lengths)), range(len(lengths)))
df_matrix = pd.DataFrame(matrix_share, range(len(lengths)), range(len(lengths)))
df_comp = pd.DataFrame(comp_share, range(len(lengths)), range(len(lengths)))

max_val = np.max([np.max(original_share), np.max(matrix_share), np.max(comp_share)])

axs[0] = sn.heatmap(df_original, annot=True, annot_kws={"size": 8}, ax=axs[0], vmax=max_val) # font size
axs[1] = sn.heatmap(df_matrix, annot=True, annot_kws={"size": 8}, ax=axs[1], vmax=max_val) # font size
axs[2] = sn.heatmap(df_comp, annot=True, annot_kws={"size": 8}, ax=axs[2], vmax=max_val) # font size
# plt.title(experiment_label)
plt.show()

print("done")
