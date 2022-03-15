import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import numpy as np
import os
import seaborn as sns
sns.set_theme(style="whitegrid")

top_dir = './data/'

# collect = 'best_testing_accuracy'
# collect = 'fold_testing_accuracy'
# collect = 'running_synapse_counts'
collect = 'running_neuron_counts'
# collect = 'epoch error'
# collect = 'pen'
# collect = 'train_test'

print(collect)
combine_plots = True
plot_final = True
reverse_variables = 1

examine = 'random'

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
    reverse_variables = 1
# variables = ['', '2']
if reverse_variables:
    variables.reverse()

colours = pl.cm.coolwarm(np.linspace(0, 1, len(variables)))

all_data = []
all_average = []
all_stderr = []

if combine_plots:
    plt.figure()
    figure = plt.gcf()
    figure.set_size_inches(16, 9)
    plt.tight_layout(rect=[0, 0.3, 1, 0.95])
    # plt.suptitle(test_label, fontsize=16)
for idx, variable in enumerate(variables):
    if examine == 'random':
        base_file_name = 'paper_var no_exp - 1ms{} sm0.0 RL0.9999  - ' \
                         'wine fixed_h0.0 - sw0.4n0.0 - at0.0 - et0.2 - 1.0adr1.0 - 0.0noise'.format(variable)
    if examine == 'surprise':
        base_file_name = 'paper_var {} - 1ms50 sm0.0 RL0.9999  - ' \
                         'wine fixed_h0.0 - sw0.4n0.0 - at0.0 - et0.0 - 1.0adr1.0 - 0.0noise'.format(variable)
    if examine == 'error':
        base_file_name = 'paper_var 0.0 - 1ms50 sm0.0 RL0.9999  - ' \
                         'wine fixed_h0.0 - sw0.4n0.0 - at0.0 - et{} - 1.0adr1.0 - 0.0noise'.format(variable)
    if examine == 'spread':
        base_file_name = 'paper_var 0.1 - 1ms50 sm0.0 RL0.9999  - ' \
                         'wine fixed_h0.0 - sw{}n0.0 - at0.0 - et0.2 - 1.0adr1.0 - 0.0noise'.format(variable)
    print(base_file_name)

    all_data_names = []
    for root, dirs, files in os.walk(top_dir):
        for file in files:
            if base_file_name in file:
                all_data_names.append(top_dir + file)
    for data_name in all_data_names:
        all_data.append((np.load(data_name, allow_pickle=True)).item())

    selected_data = all_data[-1][collect]
    average = []
    std_err = []
    min_length = min([len(run) for run in selected_data])
    for i in range(min_length):
        time_slice = []
        for j in range(len(selected_data)):
            time_slice.append(selected_data[j][i])
        average.append(np.average(time_slice))
        std_err.append(np.std(time_slice) / np.sqrt(len(selected_data)))
    all_average.append(average)
    all_stderr.append(std_err)

    if not plot_final:
        plt.xlabel('Training examples', fontsize=14)
        x = [i for i in range(min_length)]
        std_err1 = np.array(average) + np.array(std_err)
        std_err2 = np.array(average) - np.array(std_err)
        # plt.fill_between(x, std_err1, std_err2, color=colours[idx], alpha=0.5)
        plt.plot(x, average, label='{}'.format(variable), color=colours[idx])

if plot_final:
    params = {'mathtext.default': 'regular'}
    plt.rcParams.update(params)
    if examine == 'random':
        plt.xlabel('Random input sample size', fontsize=14)
    if examine == 'surprise':
        plt.xlabel('Surprise threshold ($s_{th}$)', fontsize=14)
    if examine == 'error':
        plt.xlabel('Error threshold ($e_{th}$)', fontsize=14)
    if examine == 'spread':
        plt.xlabel('Kernel spread ($s$)', fontsize=14)
    all_final_ave = [all_average[i][-1] for i in range(len(all_average))]
    all_final_std = [all_stderr[i][-1] for i in range(len(all_stderr))]
    print(all_final_ave)
    plt.errorbar(variables, all_final_ave, yerr=all_final_std, ecolor='k', capsize=3, linestyle=' ')
    plt.plot(variables, all_final_ave, color='k')
else:
    plt.legend(loc="lower right")

if 'test' in collect:
    plt.ylabel('Testing accuracy', fontsize=14)
if 'neuron' in collect:
    plt.ylabel('Neuron count', fontsize=14)
if 'synapse' in collect:
    plt.ylabel('Synapse count', fontsize=14)
plt.show()

print("done")