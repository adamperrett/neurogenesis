import matplotlib.pyplot as plt
import numpy as np
import os

top_dir = './data/'

# collect = 'fold_test_accuracy'
# collect = 'best_testing_accuracy'
# collect = 'noise_results'
# collect = 'neuron_counts'
collect = 'epoch error'
print(collect)

variables = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]#, 1., 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]
# variables = [0, 1, 2, 3, 4, 8, 15, 25, 50]
# variables = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4]
# variables = [0.000001, 0.000005, 0.0000075, 0.00001, 0.000025,
#              0.00005, 0.000075, 0.0001, 0.00025, 0.0005, 0.00075, 0.001]
# variables = [0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2]
# variables = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

for variable in variables:
    # base_file_name = 'kfold-strat always add out RL0.99999  - ' \
    #                  'breast fixed_h1000 - sw{} - at0.0 - et0.0 - 1.0adr1.0 - 0.0noise'.format(variable)
    base_file_name = 'kfold-strat save act noshuff allin out0.0 RL0.99999  - ' \
                     'wine fixed_h0 - sw{}n0.0 - at0.0 - et0.0 - 1.0adr1.0 - 0.0noise'.format(variable)
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

    collected_data = []
    time_length = []
    for data in all_data:
        if collect == 'epoch error':
            collected_data.append([data[collect][i][1] for i in range(len(data[collect]))])
        elif collect == 'noise_results':
            collected_data.append(data[collect][-1])
        else:
            collected_data.append(data[collect])
        time_length.append(len(collected_data[-1]))
        # if len(collected_data) > 1:
        #     if len(collected_data[-1]) != len(collected_data[-2]):
        #         del collected_data[-1]


    average_data = []
    std_dev_data = []
    std_err_data = []
    max_data = []
    for i in range(min(time_length)):
        time_slice = []
        for j in range(len(collected_data)):
            try:
                time_slice.append(collected_data[j][i])
            except:
                None
        max_data.append(max(time_slice))
        average_data.append(np.average(time_slice))
        std_dev_data.append(np.std(time_slice))
        std_err_data.append(np.std(time_slice) / np.sqrt(j))

    print(variable, ": ", average_data[-1], "+-", std_dev_data[-1]
          , "(+-", std_err_data[-1], ")", len(collected_data), max_data[-1], time_length)
    # for i in range(len(collected_data)):
    #     print(average_data[-1])


    plt.figure()
    # plt.plot([i for i in range(len(average_data))], np.array(average_data) + np.array(std_err_data), 'r')
    # plt.plot([i for i in range(len(average_data))], np.array(average_data) - np.array(std_err_data), 'r')
    # plt.show()
    if collect == 'noise_results':
        plt.plot(np.linspace(0, .9, 21), np.array(average_data) + np.array(std_dev_data), 'r')
        plt.plot(np.linspace(0, .9, 21), np.array(average_data) - np.array(std_dev_data), 'r')
        plt.plot(np.linspace(0, .9, 21), np.array(average_data) + np.array(std_err_data), 'g')
        plt.plot(np.linspace(0, .9, 21), np.array(average_data) - np.array(std_err_data), 'g')
        plt.plot(np.linspace(0, .9, 21), average_data, 'b')
        plt.ylim([-0.2, 1.2])
    else:
        if 'accuracy' in collect:
            plt.plot([i for i in range(len(average_data))], np.ones_like(average_data), 'k--')
        plt.plot([i for i in range(len(average_data))], np.array(average_data) + np.array(std_dev_data), 'r')
        plt.plot([i for i in range(len(average_data))], np.array(average_data) - np.array(std_dev_data), 'r')
        plt.plot([i for i in range(len(average_data))], np.array(average_data) + np.array(std_err_data), 'g')
        plt.plot([i for i in range(len(average_data))], np.array(average_data) - np.array(std_err_data), 'g')
        plt.plot([i for i in range(len(average_data))], average_data, 'b')
        if collect != 'neuron_counts':
            plt.ylim([-0.2, 1.2])
    plt.savefig("./plots/averaged {} {}.png".format(collect, base_file_name), bbox_inches='tight', dpi=200)
    plt.close()
print("done")