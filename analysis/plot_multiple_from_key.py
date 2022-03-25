import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import numpy as np
import os
import seaborn as sns
sns.set_theme(style="whitegrid")

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def compare_strings(sa, sb):
    difference = '-'
    for a, b in zip(sa, sb):
        if a != b:
            difference += b
    return difference

top_dir = '../tests/data/'
# top_dir = 'C:/Users/adam_/OneDrive/Documents/PhD/neuro_genesis' \
#           '/plots/stage_27_surprise_threshold_sands_mnist/data/'

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
    ['training_classifications', 1],
    # ['running_error_values', 1],
    # ['fold_testing_accuracy', 0],
]

print(0, '\n', collect_multiple)

combine_plots = True
pen = True
plot_final = 0
reverse_variables = 1
average_window = 100

string_keys = [
    # 'err-threshold',
    # 'sm_threshold',
    # 'rand_select'
    'bp actor critic invpen gamma0.99 - hidden128 - lr0.003',
    'paper no_act pl0.5 long2000 w10 mem0.0 RL0.9999 net300x4  - pen fixed_h0.0 - sw0.6 - at0.0 - et0.0 - 1.0adr1.0',
    'paper no_act pl0.5 long2000 w10 mem0.0 RL0.9999 net100000x4  - pen fixed_h0.0 - sw0.6 - at0.0 - et0.0 - 1.0adr1.0',
]

string_labels = [
    'actor-critic',
    'max_net_size',
]

colours = pl.cm.coolwarm(np.linspace(0, 1, len(string_keys)))

all_data = {k: [] for k, _ in collect_multiple}
processed_data = {k: [] for k, _ in collect_multiple}
all_stderr = {k: [] for k, _ in collect_multiple}

if combine_plots:
    fig, ax1 = plt.subplots()
    if len(collect_multiple) > 1:
        ax2 = ax1.twinx()
    figure = plt.gcf()
    # figure.set_size_inches(16, 9)
    # plt.tight_layout(rect=[0, 0.3, 1, 0.95])
    # plt.suptitle(test_label, fontsize=16)
for idx, base_file_name in enumerate(string_keys):

    print(base_file_name)
    for count, (collect, low_pass) in enumerate(collect_multiple):
        all_data_names = []
        for root, dirs, files in os.walk(top_dir):
            for file in files:
                if base_file_name in file:
                    all_data_names.append(top_dir + file)
        for data_name in all_data_names:
            # compare_strings(all_data_names[0], data_name)
            print("Loading:", data_name)
            if not pen:
                all_data[collect].append((np.load(data_name, allow_pickle=True)).item())
            else:
                all_data[collect].append((np.load(data_name, allow_pickle=True)))

        print("Collecting data")
        if not pen:
            looping = all_data[collect][-1][collect]
        else:
            looping = all_data[collect][-1][0]
            extended_data = []
            for trial in looping:
                extended_data.append(trial)
                while len(extended_data[-1]) < 2000:
                    extended_data[-1].append(500)
                extended_data[-1] = np.array(extended_data[-1])
            extended_data = np.array(extended_data)
            ave_data = np.mean(extended_data, axis=0)

        for selected_data in looping:
            if low_pass:
                data = moving_average(selected_data, average_window)
            else:
                data = selected_data
            processed_data[collect].append(data)

            if not plot_final:
                print("Plotting")
                ax1.set_xlabel('Training examples', fontsize=14)
                # plt.xlabel('Training examples', fontsize=14)
                x = [i for i in range(len(data))]
                # std_err1 = np.array(average) + np.array(std_err)
                # std_err2 = np.array(average) - np.array(std_err)
                # plt.fill_between(x, std_err1, std_err2, color=colours[idx], alpha=0.5)
                count = 0
                for c, _ in collect_multiple:
                    if c == collect:
                        break
                    count += 1
                if low_pass:
                    x = [i + average_window for i in range(len(data))]
                if not count:
                    ax1.plot(x, data, label='{}'.format(base_file_name), color=colours[idx])
                else:
                    ax2.plot(x, data, label='{}'.format(base_file_name), color=colours[idx])
                # plt.plot(x, average, label='{}'.format(variable), color=colours[idx])

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
        ax1.set_xlabel('Random input sample size', fontsize=14)
    if examine == 'surprise':
        ax1.set_xlabel('Surprise threshold ($s_{th}$)', fontsize=14)
    if examine == 'error':
        ax1.set_xlabel('Error threshold ($e_{th}$)', fontsize=14)
    if examine == 'spread':
        ax1.set_xlabel('Kernel spread ($s$)', fontsize=14)
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
    plt.legend(loc="lower right")

for idx, (collect, _) in enumerate(collect_multiple):
    if not idx:
        if 'test' in collect:
            ax1.set_ylabel('Testing accuracy', fontsize=14, color=colours[idx])
        if 'neuron' in collect:
            ax1.set_ylabel('Neuron count', fontsize=14, color=colours[idx])
        if 'synapse' in collect:
            ax1.set_ylabel('Synapse count', fontsize=14, color=colours[idx])
        if 'error' in collect:
            ax1.set_ylabel('Absolute error', fontsize=14, color=colours[idx])
            # ax1.plot([0, len(all_average[collect][-1])],
            #          [error_threshold, error_threshold], 'k--')
        if 'train' in collect:
            ax1.set_ylabel('Training accuracy', fontsize=14, color=colours[idx])

    else:
        if 'test' in collect:
            ax2.set_ylabel('Testing accuracy', fontsize=14, color=colours[idx])
        if 'neuron' in collect:
            ax2.set_ylabel('Neuron count', fontsize=14, color=colours[idx])
        if 'synapse' in collect:
            ax2.set_ylabel('Synapse count', fontsize=14, color=colours[idx])
        if 'error' in collect:
            ax2.set_ylabel('Absolute error', fontsize=14, color=colours[idx])
            # ax2.plot([0, len(all_average[collect][-1])],
            #          [error_threshold, error_threshold], 'k--')
        if 'train' in collect:
            ax2.set_ylabel('Training accuracy', fontsize=14, color=colours[idx])
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

