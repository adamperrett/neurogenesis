import numpy as np
import matplotlib.pyplot as plt

def stretch_data(data, stretch):
    new_data = []
    for d in data:
        for i in range(stretch):
            new_data.append(d)
    return new_data

def match_lengths(d1, d2):
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
tf_test_rate = 64
n_neurons = 350
tf_file = 'bp mpg n{} lr0.008 b{}.npy'.format(n_neurons, tf_test_rate)
tf_colour = [0, 0, 0]
ng_file = 'regression1ms8 square0.0 RL0.9999  - mpg fixed_h0.0 - sw0.4n9 - at0.0 - et0.0 - 1.0adr1.0 - 0.0noise 9.npy'
# ng_file = 'w_surprise0.05 - 10ms8 sm0.0 RL0.9999  - breast fixed_h0.0 - ' \
#           'sw0.6n0.0 - at0.0 - et0.0 - 1.0adr1.0 - 0.0noise.npy'
ng_test_rate = 1
ng_colour = [0, 0, 1]

tf_data = np.load(relative_directory+tf_file, allow_pickle=True).item()
ng_data = np.load(relative_directory+ng_file, allow_pickle=True).item()

tf_testing = stretch_data(tf_data['ave_test'], tf_test_rate)
ng_testing = stretch_data(ng_data['ave_data']['fold_testing_accuracy'], ng_test_rate)
tf_testing, ng_testing = match_lengths(tf_testing, ng_testing)

fig, ax = plt.subplots(1, 1)
plt.setp(ax, ylim=[0, 1])
ax.plot([i for i in range(len(tf_testing))], tf_testing, label='ANN', color=tf_colour)
ax.plot([i for i in range(len(ng_testing))], ng_testing, label='EDSAN', color=ng_colour)
if 'mpg' in tf_file:
    ax.legend(loc='upper right')
    ax.set_ylabel('Mean squared error', fontsize=14)
else:
    ax.legend(loc='lower right')
    ax.set_ylabel('Classification accuracy', fontsize=14)
ax.set_xlabel('Training examples', fontsize=14)
plt.show()

print('done')