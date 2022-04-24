import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def stretch_data(data, stretch):
    new_data = []
    for d in data:
        for i in range(stretch):
            new_data.append(d)
    return new_data

def force_batch_average_with_cap(data, batch_size, cap):
    new_data = []
    for i in range(0, len(data), batch_size):
        if i + batch_size > cap:
            end = cap
        else:
            end = i + batch_size
        new_data.append(np.average(data[i:end]))
        if end == cap:
            break
    return new_data

def match_lengths(d1, d2):
    if len(d1) == len(d2):
        return d1, d2
    max_length = np.max([len(d1), len(d2)])
    new_1 = []
    new_2 = []
    for i in range(max_length):
        if i >= len(d1):
            new_1.append(d1[-1])
        else:
            new_1.append(d1[i])
        if i >= len(d2):
            new_2.append(d2[-1])
        else:
            new_2.append(d2[i])
    return new_1, new_2

relative_directory = '../tests/data/'
n_neurons = 200
tf_test_rate = 64
# tf_test_rate = 8
# tf_file = 'bp wine n{} lr0.03 b{}.npy'.format(n_neurons, tf_test_rate)
ng_file = 'thresholded-True-oa-th0.4 retest1000 sm ms150  - ' \
          'mnist fixed_h0.0 - sw0.4 - at0.0 - et0.1 - 1.0adr0.99.png.npy'
# ng_file = 'err_surprise0.1 - 1ms8 sm0.0 RL0.9999  - ' \
#           'wine fixed_h0.0 - sw0.4n0.0 - at0.0 - et0.05 - 1.0adr1.0 - 0.0noise.png.npy'
tf_file = 'bp mnist n1024 lr0.001 b64.npy' #'bp testing mnist n1024 lr0.001 b64'
# ng_file = 'thresholded-True-oa-th0.4 retest1000 sm ms150  - ' \
#           'mnist fixed_h0.0 - sw0.4 - at0.0 - et0.1 - 1.0adr0.99.png.npy'
ng_test_rate = 1
tf_colour = [0, 0, 1]
ng_colour = [0, 0, 0]
ng2_colour = [1, 0, 0]
print('Loading data')
tf_data = np.load(relative_directory+tf_file, allow_pickle=True).item()
ng_data = np.load(relative_directory+ng_file, allow_pickle=True).item()
ng_data2 = np.load("C:/Users/adam_/OneDrive/Documents/PhD/neuro_genesis/plots/"
                   "stage_27_surprise_threshold_sands_mnist/rand select/data/"
                   "rand_select-False-err-th0.0 retest1000 sm ms150  - "
                   "mnist fixed_h0.0 - sw0.4 - at0.0 - et0.1 - 1.0adr0.99.npy", allow_pickle=True).item()

print('Formatting data')
if 'mnist' in ng_file:
    average_window = 50
    ng_testing = force_batch_average_with_cap(ng_data['training classifications'],
                                              batch_size=1, cap=60000)
    ng_testing2 = force_batch_average_with_cap(ng_data2['training classifications'],
                                              batch_size=1, cap=60000)
    original_ng_testing = ng_testing
    ng_testing = moving_average(ng_testing, average_window*tf_test_rate)
    ng_testing2 = moving_average(ng_testing2, average_window*tf_test_rate)
    tf_testing = stretch_data(tf_data['ave_test'], tf_test_rate)
    tf_testing = moving_average(tf_testing, average_window*tf_test_rate)
else:
    ng_testing = stretch_data(ng_data['ave_data']['fold_testing_accuracy'], ng_test_rate)
    tf_testing = stretch_data(tf_data['ave_test'], tf_test_rate)
    tf_testing, ng_testing = match_lengths(tf_testing, ng_testing)

print('plotting')
fig, ax = plt.subplots(1, 1)
legend_size = 28
fontsize = 28
tick_size = fontsize
# plt.setp(ax, ylim=[0, 1])

if 'mpg' in tf_file:
    ax.plot([i for i in range(len(tf_testing))], np.multiply(tf_testing, 1),
            label='GD', color=tf_colour)
    ax.plot([i for i in range(len(ng_testing))], np.multiply(ng_testing, 1),
            label='EDN', color=ng_colour)
    ax.legend(loc='upper right', prop={'size': legend_size})
    ax.set_ylabel('Mean squared error', fontsize=fontsize)
    ax.set_xlabel('Training examples', fontsize=fontsize)
elif 'mnist' in ng_file:
    ax.plot([i+(average_window*tf_test_rate) for i in range(len(tf_testing))], np.multiply(tf_testing, 100),
            label='GD', color=tf_colour)
    ax.plot([i+(average_window*tf_test_rate) for i in range(len(ng_testing))], np.multiply(ng_testing, 100),
            label='EDN with surprise selection', color=ng_colour)
    ax.plot([i+(average_window*tf_test_rate) for i in range(len(ng_testing2))], np.multiply(ng_testing2, 100),
            label='EDN with random selection', color=ng2_colour)
    ax.legend(loc='lower right', prop={'size': legend_size})
    ax.set_ylabel('Training accuracy averaged over 50 batches (%)', fontsize=fontsize)
    ax.set_xlabel('Training examples', fontsize=fontsize)
else:
    ax.plot([i for i in range(len(tf_testing))], np.multiply(tf_testing, 100),
            label='GD', color=tf_colour)
    ax.plot([i for i in range(len(ng_testing))], np.multiply(ng_testing, 100),
            label='EDN', color=ng_colour)
    ax.legend(loc='lower right', prop={'size': legend_size})
    ax.set_ylabel('Testing accuracy (%)', fontsize=fontsize)
    ax.set_xlabel('Training examples', fontsize=fontsize)
# ax.set_yscale('log')
plt.xticks(fontsize=tick_size)
plt.yticks(fontsize=tick_size)
plt.subplots_adjust(left=0.071, bottom=0.093,
                                              right=0.995, top=0.995,
                                              wspace=0.015, hspace=0)
plt.show()
print('done')