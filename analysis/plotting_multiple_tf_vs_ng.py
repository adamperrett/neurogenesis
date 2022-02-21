import numpy as np
import matplotlib.pyplot as plt

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
tf_test_rate = [1, 64]
n_neurons = [8192]
configs = [
    [1, 0.00003, 1024],
    [1, 0.0001, 1024],
    # [1, 0.0003, 1024],
    # [1, 0.001, 1024],
    # [1, 0.003, 1024],
    # [1, 0.01, 1024],
    # [1, 0.03, 1024]
    [32, 0.0001, 1024],
    [32, 0.0003, 1024],
    [32, 0.001, 1024],
    [32, 0.003, 1024],
    # [32, 0.01, 1024],
    # [32, 0.03, 1024],
    # [32, 0.1, 1024]
]
tf_files = []
tf_labels = []
stretch_size = []
tf_colour = []
max_brightness = 0.6
for rate, lr, n in configs:
    tf_files.append('bp none mpg n{} lr{} b{}.npy'.format(n, lr, rate))
    tf_labels.append('lr {} - n {} - b {}'.format(lr, n, rate))
    stretch_size.append(rate)
    tf_colour.append([max_brightness * len(tf_colour) / (len(configs) - 1)
                      for i in range(3)])
# ng_file = 'thresholded-True-oa-th0.4 retest1000 sm ms150  - ' \
#           'mnist fixed_h0.0 - sw0.4 - at0.0 - et0.1 - 1.0adr0.99.png.npy'
ng_file = 'regression_no_norm long 1ms20 square0.0 RL0.9999  - ' \
          'mpg fixed_h0.0 - sw0.4n9 - at0.0 - et0.0 - 1.0adr1.0 - 0.0noise.npy'
ng_test_rate = 1
ng_colour = [0, 0, 1]
print('Loading data')
tf_data = []
for tf_file in tf_files:
    tf_data.append(np.load(relative_directory+tf_file, allow_pickle=True).item())
ng_data = np.load(relative_directory+ng_file, allow_pickle=True).item()

print('Formatting data')
if 'mnist' in ng_file:
    average_window = 1000
    # ng_testing = force_batch_average_with_cap(ng_data['training classifications'],
    #                                           batch_size=tf_test_rate, cap=60000)
    ng_testing = ng_data['training classifications'][:60000]
    original_ng_testing = ng_testing
    ng_testing = moving_average(ng_testing, average_window)
    tf_testing = []
    for stretch, data in zip(stretch_size, tf_data):
        tf_testing.append(moving_average(stretch_data(data['ave_test'], stretch), average_window))
else:
    ng_testing, _ = match_lengths(ng_data['ave_data']['fold_testing_accuracy'], tf_data[0]['ave_test'])
    tf_testing = []
    for stretch, data in zip(stretch_size, tf_data):
        tf_testing.append(stretch_data(data['ave_test'], stretch))
    # tf_testing, ng_testing = match_lengths(tf_testing, ng_testing)

print('plotting')
fig, ax = plt.subplots(1, 1)
# plt.setp(ax, ylim=[0, 1])
for label, tf_t, tf_c in zip(tf_labels, tf_testing, tf_colour):
    ax.plot([i for i in range(len(tf_t))], tf_t, label='ANN '+label, color=tf_c)
ax.plot([i for i in range(len(ng_testing))], ng_testing, label='EDSAN', color=ng_colour)
if 'mpg' in tf_file:
    ax.legend(loc='upper right')
    ax.set_ylabel('Mean squared error', fontsize=14)
    ax.set_xlabel('Training examples', fontsize=14)
elif 'mnist' in ng_file:
    ax.legend(loc='lower right')
    ax.set_ylabel('Training accuracy', fontsize=14)
    ax.set_xlabel('Training examples', fontsize=14)
else:
    ax.legend(loc='lower right')
    ax.set_ylabel('Classification accuracy', fontsize=14)
    ax.set_xlabel('Training examples', fontsize=14)
plt.show()

print('done')