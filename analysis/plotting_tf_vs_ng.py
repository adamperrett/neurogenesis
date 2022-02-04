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
tf_test_rate = 1
n_neurons = 8192
tf_file = 'bp mnist n{} lr0.0001 b{}.npy'.format(n_neurons, tf_test_rate)
tf_colour = [0, 0, 0]
ng_file = 'thresholded-True-oa-th0.4 retest1000 sm ms150  - ' \
          'mnist fixed_h0.0 - sw0.4 - at0.0 - et0.1 - 1.0adr0.99.png.npy'
# ng_file = 'w_surprise0.05 - 10ms8 sm0.0 RL0.9999  - breast fixed_h0.0 - ' \
#           'sw0.6n0.0 - at0.0 - et0.0 - 1.0adr1.0 - 0.0noise.npy'
ng_test_rate = 1
ng_colour = [0, 0, 1]
print('Loading data')
tf_data = np.load(relative_directory+tf_file, allow_pickle=True).item()
ng_data = np.load(relative_directory+ng_file, allow_pickle=True).item()

print('Formatting data')
if 'mnist' in ng_file:
    average_window = 50
    ng_testing = force_batch_average_with_cap(ng_data['training classifications'],
                                              batch_size=tf_test_rate, cap=60000)
    original_ng_testing = ng_testing
    ng_testing = moving_average(ng_testing, average_window)
    tf_testing = moving_average(tf_data['ave_test'], average_window)
else:
    ng_testing = stretch_data(ng_data['ave_data']['fold_testing_accuracy'], ng_test_rate)
    tf_testing = stretch_data(tf_data['ave_test'], tf_test_rate)
tf_testing, ng_testing = match_lengths(tf_testing, ng_testing)

print('plotting')
fig, ax = plt.subplots(1, 1)
plt.setp(ax, ylim=[0, 1])
ax.plot([i for i in range(len(tf_testing))], tf_testing, label='ANN', color=tf_colour)
ax.plot([i for i in range(len(ng_testing))], ng_testing, label='EDSAN', color=ng_colour)
if 'mpg' in tf_file:
    ax.legend(loc='upper right')
    ax.set_ylabel('Mean squared error', fontsize=14)
    ax.set_xlabel('Training examples', fontsize=14)
elif 'mnist' in ng_file:
    ax.legend(loc='lower right')
    ax.set_ylabel('Training accuracy', fontsize=14)
    ax.set_xlabel('Batches', fontsize=14)
else:
    ax.legend(loc='lower right')
    ax.set_ylabel('Classification accuracy', fontsize=14)
    ax.set_xlabel('Training examples', fontsize=14)
plt.show()

print('done')