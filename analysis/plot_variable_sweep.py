import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import matplotlib
import numpy as np
import os
import seaborn as sns
sns.set_theme(style="whitegrid")

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

top_dir = '../tests/data/'

# collect = 'best_testing_accuracy'
# collect = 'fold_testing_accuracy'
# collect = 'running_synapse_counts'
# collect = 'running_neuron_counts'
# collect = 'epoch error'
# collect = 'pen'
# collect = 'train_test'
collect_multiple = [
    # ['running_neuron_counts', 0],
    # ['running_synapse_counts', 0],
    # ['training_classifications', 1],
    # ['running_error_values', 1],
    ['fold_testing_accuracy', 0],
]
examine = 'random'
# examine = 'surprise'
# examine = 'error'
# examine = 'spread'
# examine = 'pen'

bonus_file_name = False
# bonus_file_name = 'bp actor critic invpen gamma0.99 - hidden128 - lr0.003'
# bonus_file_name = 'paper no_act pl0.5 long1500 w10 mem0.0 RL0.9999 net100000x4  - ' \
#                   'pen fixed_h0.0 - sw0.6 - at0.0 - et0.0 - 1.0adr1.0'

print(examine, '\n', collect_multiple)

combine_plots = True
plot_final = 1
reverse_variables = 1
average_window = 100
error_threshold = 0.2

# variables = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]#, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]
# variables = [10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 5000]
# variables = [10, 50, 100, 150, 200, 300, 400, 500, 600]
# variables = [0]#, 1, 2, 3, 4, 5, 6, 7, 8, 9]#100]#6, 24, 32, 64]
# variables = [0.4, 0.3, 0.2, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003]#, 0.0001, 0.00003, 0.00001]0.9, 0.7, 0.6, 0.5,
# variables = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4]
# variables = [0.000001, 0.000005, 0.0000075, 0.00001, 0.000025,
#              0.00005, 0.000075, 0.0001, 0.00025, 0.0005, 0.00075, 0.001]
# variables = [0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2]
# variables = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
#              0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.]
if examine == 'random':
    variables = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    reverse_variables = 0
if examine == 'surprise':
    variables = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.]
    reverse_variables = 1
if examine == 'error':
    variables = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                 0.55, 0.6, 0.65, 0.7]
    reverse_variables = 1
if examine == 'spread':
    variables = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    # variables = [0.2, 0.3, 0.4, 0.6, 1., 2.0]
    reverse_variables = 1
if examine == 'pen':
    variables = [100000, 50, 100, 150, 200, 250, 300, 350]
    variable_string = {}
# variables = ['', '2']
if reverse_variables:
    variables.reverse()

colours = pl.cm.coolwarm(np.linspace(0, 1, len(variables)))
legend_size = 17
fontsize = 28
tick_size = fontsize

all_data = {k: [] for k, _ in collect_multiple}
all_average = {k: [] for k, _ in collect_multiple}
all_stderr = {k: [] for k, _ in collect_multiple}

if combine_plots:
    fig, ax1 = plt.subplots()
    if len(collect_multiple) > 1:
        ax2 = ax1.twinx()
    figure = plt.gcf()
    # figure.set_size_inches(16, 9)
    # plt.tight_layout(rect=[0, 0.3, 1, 0.95])
    # plt.suptitle(test_label, fontsize=16)
for idx, variable in enumerate(variables):
    if examine == 'random':
        base_file_name = 'paper_var no_exp - 1ms{} sm0.0 RL0.9999  - ' \
                         'wine fixed_h0.0 - sw0.4n0.0 - at0.0 - et0.2 - 1.0adr1.0 - 0.0noise'.format(variable)
    if examine == 'surprise':
        # base_file_name = 'paper_var {} - 1ms50 sm0.0 RL0.9999  - ' \
        #                  'wine fixed_h0.0 - sw0.4n0.0 - at0.0 - et0.0 - 1.0adr1.0 - 0.0noise'.format(variable)
        base_file_name = 'paper_var exp{} - 1ms1 sm0.0 RL0.9999  - ' \
                         'wine fixed_h0.0 - sw0.4n0.0 - at0.0 - et0.2 - 1.0adr1.0 - 0.0noise'.format(variable)

    if examine == 'error':
        # base_file_name = 'paper_var 0.0 - 1ms50 sm0.0 RL0.9999  - ' \
        #                  'wine fixed_h0.0 - sw0.4n0.0 - at0.0 - et{} - 1.0adr1.0 - 0.0noise'.format(variable)
        base_file_name = 'paper_var exp0.1 - 1ms1 sm0.0 RL0.9999  - ' \
                         'wine fixed_h0.0 - sw0.4n0.0 - at0.0 - et{} - 1.0adr1.0 - 0.0noise'.format(variable)

    if examine == 'spread':
        # base_file_name = 'paper_var 0.1 - 1ms50 sm0.0 RL0.9999  - ' \
        #                  'wine fixed_h0.0 - sw{}n0.0 - at0.0 - et0.2 - 1.0adr1.0 - 0.0noise'.format(variable)
        base_file_name = 'paper_var exp0.1 - 1ms1 sm0.0 RL0.9999  - ' \
                         'wine fixed_h0.0 - sw{}n0.0 - at0.0 - et0.2 - 1.0adr1.0 - 0.0noise'.format(variable)
    if examine == 'pen': #
        base_file_name = 'paper ' \
                         'no_act ' \
                         'pl0.5 long2000 w10 mem0.0 RL0.9999 ' \
                         'net{}x4  - pen fixed_h0.0 - sw0.6 - at0.0 - et0.0 - 1.0adr1.0'.format(variable)


    print(base_file_name)
    for count, (collect, low_pass) in enumerate(collect_multiple):
        all_data_names = []
        for root, dirs, files in os.walk(top_dir):
            for file in files:
                if base_file_name in file:
                    all_data_names.append(top_dir + file)
        for data_name in all_data_names:
            if examine == 'pen':
                all_data[collect].append((np.load(data_name, allow_pickle=True)))
                selected_data = all_data[collect][-1]
            else:
                all_data[collect].append((np.load(data_name, allow_pickle=True)).item())
                selected_data = all_data[collect][-1][collect]

        # selected_data = all_data[collect][-1][collect]
        average = []
        std_err = []
        min_length = min([len(run) for run in selected_data])
        for i in range(min_length):
            time_slice = []
            for j in range(len(selected_data)):
                time_slice.append(selected_data[j][i])
            average.append(np.average(time_slice))
            std_err.append(np.std(time_slice) / np.sqrt(len(selected_data)))
        if low_pass:
            average = moving_average(average, average_window)
            std_err = moving_average(std_err, average_window)
        all_average[collect].append(average)
        all_stderr[collect].append(std_err)

        if not plot_final:
            ax1.set_xlabel('Training examples', fontsize=fontsize)
            # plt.xlabel('Training examples', fontsize=fontsize)
            x = [i for i in range(min_length)]
            std_err1 = np.array(average) + np.array(std_err)
            std_err2 = np.array(average) - np.array(std_err)
            # plt.fill_between(x, std_err1, std_err2, color=colours[idx], alpha=0.5)
            count = 0
            for c, _ in collect_multiple:
                if c == collect:
                    break
                count += 1
            if low_pass:
                x = [i + average_window for i in range(min_length - average_window + 1)]
            if not count:
                if variable > 10000:
                    ax1.plot(x, average, label='no deletion', color=colours[idx])
                else:
                    ax1.plot(x, average, label='{}'.format(variable), color=colours[idx])
            else:
                ax2.plot(x, average, label='{}'.format(variable), color=colours[idx])
            # plt.plot(x, average, label='{}'.format(variable), color=colours[idx])

if bonus_file_name:
    for count, (collect, low_pass) in enumerate(collect_multiple):
        all_data_names = []
        for root, dirs, files in os.walk(top_dir):
            for file in files:
                if bonus_file_name in file:
                    all_data_names.append(top_dir + file)
        for data_name in all_data_names:
            if examine == 'pen':
                # all_data[collect].append((np.load(data_name, allow_pickle=True)))
                # selected_data = all_data[collect][-1]
                all_data[collect].append((np.load(data_name, allow_pickle=True)))
                selected_data = all_data[collect][-1][0]
                extended_data = []
                for trial in selected_data:
                    extended_data.append(trial)
                    while len(extended_data[-1]) < 2000:
                        extended_data[-1].append(500)
                selected_data = extended_data

        # selected_data = all_data[collect][-1][collect]
        average = []
        std_err = []
        min_length = min([len(run) for run in selected_data])
        for i in range(min_length):
            time_slice = []
            for j in range(len(selected_data)):
                time_slice.append(selected_data[j][i])
            average.append(np.average(time_slice))
            std_err.append(np.std(time_slice) / np.sqrt(len(selected_data)))
        if low_pass:
            average = moving_average(average, average_window)
            std_err = moving_average(std_err, average_window)
        all_average[collect].append(average)
        all_stderr[collect].append(std_err)

        if not plot_final:
            ax1.set_xlabel('Trials', fontsize=fontsize)
            # plt.xlabel('Training examples', fontsize=fontsize)
            x = [i for i in range(min_length)]
            std_err1 = np.array(average) + np.array(std_err)
            std_err2 = np.array(average) - np.array(std_err)
            # plt.fill_between(x, std_err1, std_err2, color=colours[idx], alpha=0.5)
            count = 0
            for c, _ in collect_multiple:
                if c == collect:
                    break
                count += 1
            if low_pass:
                x = [i + average_window for i in range(min_length - average_window + 1)]
            if not count:
                ax1.plot(x, average, label='actor-critic', color='k')
                ax1.plot([x[0], x[-1]], [475, 475], 'k--')

if len(collect_multiple) <= 1:
    colours = ['k']
else:
    if not plot_final:
        colours = ['k' for i in range(len(collect_multiple))]
    else:
        colours = pl.cm.coolwarm(np.linspace(0, 1, len(collect_multiple)))

if plot_final:
    params = {'mathtext.default': 'regular'}
    plt.rcParams.update(params)
    if examine == 'random':
        ax1.set_xlabel('Random input sample size', fontsize=fontsize)
    if examine == 'surprise':
        ax1.set_xlabel('Surprise threshold ($s_{th}$)', fontsize=fontsize)
    if examine == 'error':
        ax1.set_xlabel('Error threshold ($E_{th}$)', fontsize=fontsize)
    if examine == 'spread':
        ax1.set_xlabel('Kernel spread ($s$)', fontsize=fontsize)
    for idx, (collect, _) in enumerate(collect_multiple):
        all_final_ave = [all_average[collect][i][-1] for i in range(len(all_average[collect]))]
        all_final_std = [all_stderr[collect][i][-1] for i in range(len(all_stderr[collect]))]
        print(all_final_ave)
        # plt.errorbar(variables, all_final_ave, yerr=all_final_std, ecolor='k', capsize=3, linestyle=' ')
        # plt.plot(variables, all_final_ave, color='k')
        # plt.errorbar(variables, all_final_ave, yerr=all_final_std,
        #              ecolor=colours[idx], capsize=3, linestyle=' ')
        # plt.plot(variables, all_final_ave, color=colours[idx], secondary_y=idx)
        if not idx:
            ax1.errorbar(variables, all_final_ave, yerr=all_final_std,
                         ecolor=colours[idx], capsize=3, linestyle=' ')
            ax1.plot(variables, all_final_ave, color=colours[idx])
        else:
            ax2.errorbar(variables, all_final_ave, yerr=all_final_std,
                         ecolor=colours[idx], capsize=3, linestyle=' ')
            ax2.plot(variables, all_final_ave, color=colours[idx])
else:
    plt.legend(loc="lower right", prop={'size': legend_size})

for idx, (collect, _) in enumerate(collect_multiple):
    if not idx:
        if 'test' in collect:
            ax1.set_ylabel('Testing accuracy', fontsize=fontsize, color=colours[idx])
        if 'neuron' in collect:
            ax1.set_ylabel('Neuron count', fontsize=fontsize, color=colours[idx])
        if 'synapse' in collect:
            ax1.set_ylabel('Synapse count', fontsize=fontsize, color=colours[idx])
        if 'error' in collect:
            ax1.set_ylabel('Absolute error', fontsize=fontsize, color=colours[idx])
            # ax1.plot([0, len(all_average[collect][-1])],
            #          [error_threshold, error_threshold], 'k--')
        if 'train' in collect:
            # ax1.set_ylabel('Training accuracy', fontsize=fontsize, color=colours[idx])
            ax1.set_ylabel('Average balance length over last 100 trials', fontsize=fontsize, color=colours[idx])
        ax1.tick_params(labelsize=tick_size)
        ax1.tick_params(labelsize=tick_size)
        plt.subplots_adjust(left=0.07, bottom=0.095,
                            right=0.995, top=0.995,
                            wspace=0.015, hspace=0)
    else:
        if 'test' in collect:
            ax2.set_ylabel('Testing accuracy', fontsize=fontsize, color=colours[idx])
        if 'neuron' in collect:
            ax2.set_ylabel('Neuron count', fontsize=fontsize, color=colours[idx])
        if 'synapse' in collect:
            ax2.set_ylabel('Synapse count', fontsize=fontsize, color=colours[idx])
        if 'error' in collect:
            ax2.set_ylabel('Absolute error', fontsize=fontsize, color=colours[idx])
            # ax2.plot([0, len(all_average[collect][-1])],
            #          [error_threshold, error_threshold], 'k--')
        if 'train' in collect:
            ax2.set_ylabel('Training accuracy', fontsize=fontsize, color=colours[idx])
        ax2.tick_params(labelsize=tick_size)
        ax2.tick_params(labelsize=tick_size)
        plt.subplots_adjust(left=0.065, bottom=0.102,
                            right=0.92, top=0.98,
                            wspace=0.015, hspace=0)

plt.show()

for d in all_data[collect]:
    print('\n', d['ave_data']['epoch_error'][0])
    print(d['ave_data']['epoch_error'][-1])
    # for i in range(len(data['training_classifications'])):
    #     print(data['training_classifications'][i])
    combined = [np.average(d['training_classifications'][i]) for i in range(len(d['training_classifications']))]
    print(np.average(combined), combined)

for d in all_data[collect]:
    print('\n', d['ave_data']['epoch_error'][0])
    print(d['ave_data']['epoch_error'][-1])
    # for i in range(len(data['training_classifications'])):
    #     print(data['training_classifications'][i])
    combined = [np.average([dt > 0.2 for dt in d['running_error_values'][i]])
                for i in range(len(d['running_error_values']))]
    print(np.average(combined), combined)

print("done")

