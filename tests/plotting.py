import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import numpy as np
import os

top_dir = './data/'

# collect = 'best_testing_accuracy'
# collect = 'fold_test_accuracy'
# collect = 'noise_results'
# collect = 'neuron_counts'
collect = 'epoch error'
# collect = 'pen'
print(collect)
combine_plots = True

# pen_label = 'pl0.5 - trans random long400 w10 mem0.0 RL0.9999 net150x10  - pen fixed_h0.0 - sw0.6 - at0.0 - et0.0 - 1.0adr1.0.png'
# pendulum_data = np.load('./data/'+pen_label+'.npy', allow_pickle=True)#'.item()

# variables = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]#, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]
# variables = [0, 1, 2, 4, 8, 16, 24, 32, 64]
variables = [0.4, 0.3, 0.2, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003]#, 0.0001, 0.00003, 0.00001]0.9, 0.7, 0.6, 0.5,
# variables = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4]
# variables = [0.000001, 0.000005, 0.0000075, 0.00001, 0.000025,
#              0.00005, 0.000075, 0.0001, 0.00025, 0.0005, 0.00075, 0.001]
# variables = [0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2]
# variables = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
# variables = [1]
colours = pl.cm.GnBu(np.linspace(0, 1, len(variables)))

if combine_plots:
    plt.figure()
for idx, variable in enumerate(variables):
    # base_file_name = 'kfold-strat always add out RL0.99999  - ' \
    #                  'breast fixed_h1000 - sw{} - at0.0 - et0.0 - 1.0adr1.0 - 0.0noise'.format(variable)
    base_file_name = 'noOut thresholded-add1.0 sm0.0 RL0.99999  - ' \
                     'breast fixed_h0 - sw0.6n0.0 - at0.0 - et{} - 1.0adr1.0 - 0.0noise'.format(variable)
    print(base_file_name)
    # base_file_name = pen_label
    # base_file_name = 'withoutOUT out RL0.99999 net3000000x100  - wine fixed_h50 - ' \
    # 'sw{} - at0.0 - et0.0 - 1.0adr1.0'.format(variable)

    all_data_names = []
    for root, dirs, files in os.walk(top_dir):
        for file in files:
            if base_file_name in file:
                all_data_names.append(top_dir + file)
    all_data = []
    for data_name in all_data_names:
        all_data.append((np.load(data_name, allow_pickle=True)).item())

    time_length = []
    collected_data = []
    if collect == 'pen':
        def moving_average(a, n=3):
            ret = np.cumsum(a, dtype=float)
            ret[n:] = ret[n:] - ret[:-n]
            return ret[n - 1:] / n
        for data in all_data[0]:
            collected_data.append(moving_average(data, n=100))
            time_length.append(len(collected_data[-1]))
    else:
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
    if collect == 'epoch error':
        print([round(ad, 3) for ad in average_data])

    print(variable, ": ", average_data[-1], "+-", std_dev_data[-1]
          , "(+-", std_err_data[-1], ")", len(collected_data), max_data[-1], time_length
          , "\n", [round(ts, 3) for ts in time_slice], "\n")
    # for i in range(len(collected_data)):
    #     print(average_data[-1])

    if not combine_plots:
        plt.figure()
    # plt.plot([i for i in range(len(average_data))], np.array(average_data) + np.array(std_err_data), 'r')
    # plt.plot([i for i in range(len(average_data))], np.array(average_data) - np.array(std_err_data), 'r')
    # plt.show()
    if collect == 'noise_results':
        # plt.plot(np.linspace(0, .9, 21), np.array(average_data) + np.array(std_dev_data), 'r')
        # plt.plot(np.linspace(0, .9, 21), np.array(average_data) - np.array(std_dev_data), 'r')
        # plt.plot(np.linspace(0, .9, 21), np.array(average_data) + np.array(std_err_data), 'g')
        # plt.plot(np.linspace(0, .9, 21), np.array(average_data) - np.array(std_err_data), 'g')
        x = np.linspace(0, .9, 21)
        stdev1 = np.array(average_data) + np.array(std_err_data)
        stdev2 = np.array(average_data) - np.array(std_err_data)
        plt.fill_between(x, stdev1, stdev2, color=colours[idx], alpha=0.5)
        plt.plot(x, average_data, color=colours[idx], label='{}'.format(variable))
        plt.legend(loc="upper right")
        plt.ylim([-0.2, 1.2])
        plt.ylabel('Test accuracy')
        plt.xlabel('Standard deviation of noise')
    else:
        if 'accuracy' in collect:
            plt.plot([i for i in range(len(average_data))], np.ones_like(average_data), 'k--')
        # plt.plot([i for i in range(len(average_data))], np.array(average_data) + np.array(std_dev_data), 'r')
        # plt.plot([i for i in range(len(average_data))], np.array(average_data) - np.array(std_dev_data), 'r')
        if collect == 'pen':
            x = [i for i in range(100, len(average_data) + 100)]
            long_x = [i for i in range(400)]
            plt.plot(long_x, np.ones_like(long_x)*475, 'k--')
        else:
            x = [i for i in range(len(average_data))]
        # stdev1 = np.array(average_data) + np.array(std_dev_data)
        # stdev2 = np.array(average_data) - np.array(std_dev_data)
        stdev1 = np.array(average_data) + np.array(std_err_data)
        stdev2 = np.array(average_data) - np.array(std_err_data)
        # plt.plot(x, stdev1, 'g')
        # plt.plot(x, stdev2, 'g')
        plt.fill_between(x, stdev1, stdev2, color=colours[idx], alpha=0.5)
        plt.plot(x, average_data, color=colours[idx], label='{}'.format(variable))
        plt.legend(loc="upper right")
        if collect != 'neuron_counts':
            if collect == 'pen':
                for data in collected_data:
                    plt.plot(x, data, color='red', alpha=0.25)
                plt.xlim([0, 400])
                plt.ylim([-0.2, 510.2])
                plt.ylabel('Time steps balanced')
                plt.xlabel('Number of trials')
            else:
                plt.ylim([-0.2, 1.2])
    if not combine_plots:
        plt.savefig("./plots/averaged {} {}.png".format(collect, base_file_name), bbox_inches='tight', dpi=200)
        plt.close()
if combine_plots:
    plt.savefig("./plots/averaged {} {}.png".format(collect, base_file_name), bbox_inches='tight', dpi=200)
    plt.close()
print("done")