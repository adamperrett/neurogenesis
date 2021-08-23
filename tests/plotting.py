import matplotlib.pyplot as plt
import numpy as np
import os

top_dir = './data/'
variables = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for variable in variables:
    base_file_name = 'incremental random list only improve out RL0.99999 net300000x1000  - ' \
                     'breast fixed_h1000 - sw{} - at0.0 - et0.0 - 1.0adr1.0'.format(variable)
    # base_file_name = 'withoutOUT out RL0.99999 net3000000x100  - wine fixed_h50 - ' \
    # 'sw{} - at0.0 - et0.0 - 1.0adr1.0'.format(variable)

    all_data_names = []
    for root, dirs, files in os.walk(top_dir):
        for file in files:
            if base_file_name in file:
                all_data_names.append(top_dir + file)
    all_data = []
    for data_name in all_data_names:
        all_data.append(np.load(data_name, allow_pickle=True).item())

    # collect = 'fold_test_accuracy'
    collect = 'best_testing_accuracy'
    # collect = 'neuron_counts'
    collected_data = []
    for data in all_data:
        collected_data.append(data[collect])
        if len(collected_data) > 1:
            if len(collected_data[-1]) != len(collected_data[-2]):
                del collected_data[-1]


    average_data = []
    std_dev_data = []
    std_err_data = []
    for i in range(len(collected_data[0])):
        time_slice = []
        for j in range(len(collected_data)):
            time_slice.append(collected_data[j][i])
        average_data.append(np.average(time_slice))
        std_dev_data.append(np.std(time_slice))
        std_err_data.append(np.std(time_slice) / j)

    print(variable, ": ", average_data[-1], "+-", std_dev_data[-1])

    plt.figure()
    plt.plot([i for i in range(len(average_data))], average_data, 'b')
    plt.plot([i for i in range(len(average_data))], np.array(average_data) + np.array(std_dev_data), 'r')
    plt.plot([i for i in range(len(average_data))], np.array(average_data) - np.array(std_dev_data), 'r')
    # plt.plot([i for i in range(len(average_data))], np.array(average_data) + np.array(std_err_data), 'r')
    # plt.plot([i for i in range(len(average_data))], np.array(average_data) - np.array(std_err_data), 'r')
    # plt.show()
    plt.savefig("./plots/averaged {} {}.png".format(collect, base_file_name), bbox_inches='tight', dpi=200)
    plt.close()
print("done")