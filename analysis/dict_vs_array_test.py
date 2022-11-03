import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GaussianNoise
from streamlit import legacy_caching as caching

def original(short_dict, long_dict):
    times = []
    print("Doing original")
    for r in range(repeats):
        all_values = []
        t0 = time.time_ns()
        for i in range(neurons):
            values = 0.
            for ele in short_dict[i]:
                if ele in long_dict:
                    values += np.max([0, 1 - np.abs((short_dict[i][ele] - long_dict[ele]) / 0.6)])
            all_values.append(values)
        t1 = time.time_ns()
        times.append(t1 - t0)
    print(all_values[0])
    return times

def tf_math(short_dict, long_dict):
    # num_inputs = len(short_dict)
    # n_neurons = len(long_dict)
    # model = Sequential()
    # # model.add(GaussianNoise(stddev=k_stdev))
    # model.add(Dense(n_neurons, input_dim=num_inputs, activation='relu'))
    # # model.add(Dense(n_neurons, input_dim=num_inputs, activation='relu'))
    # if test == 'mpg':
    #     model.add(Dense(num_outputs))
    #     # model.add(Dense(num_outputs, activation='sigmoid'))
    #     # model.add(Dense(num_outputs, activation='linear'))
    # else:
    #     model.add(Dense(num_outputs, activation='softmax'))

    times = []
    print("Doing tensorflow")
    for r in range(repeats):
        short_long = []
        long_values = None
        for i in range(neurons):
            short_neurons = []
            for ele in long_dict:
                if ele in short_dict[i]:
                    short_neurons.append(short_dict[i][ele])
                else:
                    short_neurons.append(np.nan)
            short_long.append(np.array(short_neurons))
        short_long = np.array(short_long)
        long_values = np.array(list(long_dict.values()))

        # def tf_triangle(s_tf, l_tf):
        #     return tf.keras.metrics.Sum(tf.maximum(0, 0.6 - tf.abs(tf.subtract(s_tf, l_tf))), dtype=tf.float32) / 0.6

        width = tf.constant(0.6, dtype=tf.float64)
        t0 = time.time_ns()

        # short_tf = tf.constant(short_long)
        # long_tf = tf.constant(long_values)
        # av_s = tf.subtract(long_values, short_long)
        # av_a = tf.abs(tf.subtract(long_values, short_long))
        # av_w = tf.subtract(width, tf.abs(tf.subtract(long_values, short_long)))
        # av_m = tf.maximum(0, tf.subtract(width, tf.abs(tf.subtract(long_values, short_long))))
        # av_r = tf.experimental.numpy.nansum(tf.maximum(0, tf.subtract(width, tf.abs(tf.subtract(long_values, short_long)))), axis=1)

        # all_values = tf.divide(tf.reduce_sum(tf.maximum(0, tf.subtract(width, tf.abs(tf.subtract(short_tf, long_tf))))), width)
        all_values = tf.divide(tf.experimental.numpy.nansum(tf.maximum(
            0, tf.subtract(width, tf.abs(tf.subtract(long_values, short_long)))), axis=1), width)

        # print("tf\n", "\ns", av_s.numpy(), "\na", av_a.numpy(), "\nw", av_w.numpy(),
        #       "\nm", av_m.numpy(), "\nr", av_r, "\na", all_values.numpy())


        # all_values = np.sum(np.maximum(0, 0.6 - np.abs(np.subtract(short_long, long_values))), axis=1) / 0.6
        t1 = time.time_ns()
        times.append((t1 - t0) / neurons)
    # print(np.maximum(0, 0.6 - np.abs(np.subtract(short_long, long_values))) / 0.6)
    print(all_values[0].numpy())
    return times

def numpy_matrix_full(short_dict, long_dict):
    times = []
    print("Doing full matrix")
    for r in range(repeats):
        caching.clear_cache()
        short_long = []
        long_values = None
        for i in range(neurons):
            short_neurons = []
            for ele in long_dict:
                if ele in short_dict[i]:
                    short_neurons.append(short_dict[i][ele])
                else:
                    short_neurons.append(np.nan)
            short_long.append(np.array(short_neurons))
        short_long = np.array(short_long)
        long_values = np.array(list(long_dict.values()))

        # av_s = np.subtract(short_long, long_values)
        # av_a = np.abs(np.subtract(short_long, long_values))
        # av_w = 0.6 - np.abs(np.subtract(short_long, long_values))
        # av_m = np.maximum(0, 0.6 - np.abs(np.subtract(short_long, long_values)))
        # av_r = np.nansum(np.maximum(0, 0.6 - np.abs(np.subtract(short_long, long_values))), axis=1)

        t0 = time.time_ns()

        all_values = np.nansum(np.maximum(0, 0.6 - np.abs(np.subtract(short_long, long_values))), axis=1) / 0.6

        # print("tf\n", "\ns", av_s, "\na", av_a, "\nw", av_w, "\nm", av_m, "\nr", av_r, "\na", all_values)

        t1 = time.time_ns()
        times.append((t1 - t0) / neurons)
    # print(np.maximum(0, 0.6 - np.abs(np.subtract(short_long, long_values))) / 0.6)
    print(all_values[0])
    return times

def numpy_matrix_short(short_dict, long_dict):
    times = []
    print("Doing short matrix")
    for r in range(repeats):
        caching.clear_cache()
        short_values = np.array([np.array(list(short_dict[i].values())) for i in range(neurons)])
        long_values = None
        short_long = []

        t0 = time.time_ns()
        for i in range(neurons):
            short_neurons = []
            for ele in short_dict[i]:
                if ele in long_dict:
                    short_neurons.append(long_dict[ele])
            short_long.append(np.array(short_neurons))
        long_values = np.array(short_long)

        all_values = np.sum(np.maximum(0, 0.6 - np.abs(np.subtract(short_values, long_values))), axis=1) / 0.6

        t1 = time.time_ns()
        times.append((t1 - t0) / neurons)
    # print(np.maximum(0, 0.6 - np.abs(np.subtract(short_long, long_values))) / 0.6)
    print(all_values[0])
    return times

def numpy_matrix_pointer(short_dict, long_dict):
    times = []
    print("Doing short matrix")
    for r in range(repeats):
        caching.clear_cache()
        short_values = np.array([np.array(list(short_dict[i].values())) for i in range(neurons)])
        long_values = None
        short_long = []

        long_values = np.array(list(long_dict.values()))
        long_values.reshape(long_values, [len(long_values), 1])
        for i in range(neurons):
            short_neurons = []
            for ele in short_dict[i]:
                if ele in long_dict:
                    short_neurons.append(long_dict[ele])
            short_long.append(np.array(short_neurons))
        long_values = np.array(short_long)
        t0 = time.time_ns()

        all_values = np.sum(np.maximum(0, 0.6 - np.abs(np.subtract(short_values, long_values))), axis=1) / 0.6

        t1 = time.time_ns()
        times.append((t1 - t0) / neurons)
    # print(np.maximum(0, 0.6 - np.abs(np.subtract(short_long, long_values))) / 0.6)
    print(all_values[0])
    return times

def sample_from_indexes(short_dict, long_dict):
    times = []
    print("Doing sample inside")
    short_values = np.array([np.array(list(short_dict[i].values())) for i in range(neurons)])
    long_values = np.array(list(long_dict.values()))
    short_indexes = []
    long_indexes = []
    for i in range(neurons):
        neurons_short = []
        neurons_long = []
        for idx, ele in enumerate(long_dict):
            if ele in short_dict[0]:
                neurons_short.append(idx)
            else:
                neurons_long.append(idx)
        short_indexes.append(neurons_short)
        long_indexes.append(neurons_long)
    for r in range(repeats):
        all_values = []
        t0 = time.time_ns()
        for i in range(neurons):

            # all_values = 0.
            selected_long = long_values[short_indexes[i]]
            # selected_long = np.delete(long_values, long_indexes[i])
            # all_values = np.sum(np.maximum(0, 1 - np.abs(np.subtract(short_values, selected_long) / 0.6)))
            all_values.append(np.sum(np.maximum(0, 0.6 - np.abs(np.subtract(short_values[i], selected_long)))) / 0.6)

        t1 = time.time_ns()
        times.append(t1 - t0)
    print(all_values[0])
    return times

def delete_from_indexes(short_dict, long_dict):
    times = []
    print("Doing delete inside")
    short_values = np.array([np.array(list(short_dict[i].values())) for i in range(neurons)])
    long_values = np.array(list(long_dict.values()))
    short_indexes = []
    long_indexes = []
    for i in range(neurons):
        neurons_short = []
        neurons_long = []
        for idx, ele in enumerate(long_dict):
            if ele in short_dict[0]:
                neurons_short.append(idx)
            else:
                neurons_long.append(idx)
        short_indexes.append(neurons_short)
        long_indexes.append(neurons_long)
    for r in range(repeats):
        all_values = []
        t0 = time.time_ns()
        for i in range(neurons):

            # all_values = 0.
            # selected_long = long_values[short_indexes[i]]
            selected_long = np.delete(long_values, long_indexes[i])
            # all_values = np.sum(np.maximum(0, 1 - np.abs(np.subtract(short_values, selected_long) / 0.6)))
            all_values.append(np.sum(np.maximum(0, 0.6 - np.abs(np.subtract(short_values[i], selected_long)))) / 0.6)

        t1 = time.time_ns()
        times.append(t1 - t0)
    print(all_values[0])
    return times

def create_short_list_outside(short_dict, long_dict):
    times = []
    print("Doing create short list outside")
    short_values = np.array([np.array(list(short_dict[i].values())) for i in range(neurons)])
    for r in range(repeats):
        all_values = []
        t0 = time.time_ns()
        for i in range(neurons):
            long_values = {k: long_dict[k] for k in short_dict[i]}
            long_values = np.array(list(long_values.values()))
            # all_values = np.sum([1 - np.abs((short_dict[ele] - long_dict[ele]) / 0.6) for ele in short_dict])

            # all_values = 0.
            # all_values = np.sum(np.maximum(0, 1 - np.abs(np.subtract(short_values, long_values) / 0.6)))
            all_values.append(np.sum(np.maximum(0, 0.6 - np.abs(np.subtract(short_values[i], long_values)))) / 0.6)

        t1 = time.time_ns()
        times.append(t1 - t0)
    print(all_values[0])
    return times

def create_long_list_outside(short_dict, long_dict):
    times = []
    print("Doing initialise outside as long list")
    short_long = []
    for i in range(neurons):
        short_neurons = []
        for ele in long_dict:
            if ele in short_dict[i]:
                short_neurons.append(short_dict[i][ele])
            else:
                short_neurons.append(2)
        short_long.append(np.array(short_neurons))
    short_long = np.array(short_long)
    long_values = np.array(list(long_dict.values()))
    for r in range(repeats):
        all_values = []
        t0 = time.time_ns()
        for i in range(neurons):
            # all_values = np.sum([1 - np.abs((short_dict[ele] - long_dict[ele]) / 0.6) for ele in short_dict])

            # all_values = 0.
            # all_values = np.sum(np.maximum(0, 1 - np.abs(np.subtract(short_long, long_values) / 0.6)))
            all_values.append(np.sum(np.maximum(0, 0.6 - np.abs(np.subtract(short_long[i], long_values)))) / 0.6)

        t1 = time.time_ns()
        times.append(t1 - t0)
    print(all_values[0])
    return times

def test_calc_times(short_length, long_length):

    long_dict = {}
    for i in range(long_length):
        long_dict['{}'.format(i)] = np.random.random()
    short_dict = [{} for i in range(neurons)]
    for i in range(neurons):
        selection = np.random.choice([k for k in long_dict.keys()], short_length, replace=False)
        for j in range(short_length):
            # short_dict['{}'.format(i+(long_length/2))] = 100 - i
            short_dict[i]['{}'.format(selection[j])] = np.random.random()

    print("\n\n")
    time_original = numpy_matrix_short(short_dict, long_dict)
    # time_original = original(short_dict, long_dict)

    time_matrix = numpy_matrix_full(short_dict, long_dict)

    time_sample = sample_from_indexes(short_dict, long_dict)

    # time_delete = numpy_matrix_short(short_dict, long_dict)
    time_delete = delete_from_indexes(short_dict, long_dict)

    # time_pre = numpy_matrix_short(short_dict, long_dict)
    time_pre = create_short_list_outside(short_dict, long_dict)

    # time_all = create_long_list_outside(short_dict, long_dict)
    time_all = tf_math(short_dict, long_dict)
    print("")

    # total_time = np.sum([np.average(time_original),
    #                      np.average(time_matrix),
    #                      np.average(time_sample),
    #                      np.average(time_delete),
    #                      np.average(time_pre),
    #                      np.average(time_all)])
    total_time = np.max([np.min([
        # np.average(time_original),
        # np.average(time_matrix),
        np.average(time_sample),
        np.average(time_delete),
        np.average(time_pre),
        np.average(time_all)
    ]), 0.000001])
    print("with short length", short_length, "and long length", long_length)
    print(np.average(time_original) / total_time, "time for original calc = ", np.average(time_original), "with sdtev", np.std(time_original))
    print(np.average(time_matrix) / total_time, "time for np matrix = ", np.average(time_matrix), "with sdtev", np.std(time_matrix))
    print(np.average(time_sample) / total_time, "time for inside sample = ", np.average(time_sample), "with sdtev", np.std(time_sample))
    print(np.average(time_delete) / total_time, "time for inside delete = ", np.average(time_delete), "with sdtev", np.std(time_delete))
    print(np.average(time_pre) / total_time, "time for pre shorten = ", np.average(time_pre), "with sdtev", np.std(time_pre))
    print(np.average(time_all) / total_time, "time for tf math = ", np.average(time_all), "with sdtev", np.std(time_all), "\n")

    return np.average(time_original) / total_time, np.average(time_original), \
           np.average(time_matrix) / total_time, np.average(time_matrix), \
           np.average(time_sample) / total_time, np.average(time_sample), \
           np.average(time_delete) / total_time, np.average(time_delete), \
           np.average(time_pre) / total_time, np.average(time_pre), \
           np.average(time_all) / total_time, np.average(time_all)

lengths = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

neurons = 10000
repeats = 10

all_data = []
original_share = [[0 for i in range(len(lengths))] for j in range(len(lengths))]
matrix_share = [[0 for i in range(len(lengths))] for j in range(len(lengths))]
sample_share = [[0 for i in range(len(lengths))] for j in range(len(lengths))]
delete_share = [[0 for i in range(len(lengths))] for j in range(len(lengths))]
short_share = [[0 for i in range(len(lengths))] for j in range(len(lengths))]
long_share = [[0 for i in range(len(lengths))] for j in range(len(lengths))]
for x, sl in enumerate(lengths):
    for y, ll in enumerate(lengths):
        if sl <= ll:
            result = test_calc_times(sl, ll)
            all_data.append(result)
            # original_share[x][y] = result[0]
            # matrix_share[x][y] = result[2]
            # sample_share[x][y] = result[4]
            # delete_share[x][y] = result[6]
            # short_share[x][y] = result[8]
            # long_share[x][y] = result[10]
            # original_share[x][y] = result[1]
            # matrix_share[x][y] = result[3]
            # sample_share[x][y] = result[5]
            # delete_share[x][y] = result[7]
            # short_share[x][y] = result[9]
            # long_share[x][y] = result[11]
            original_share[x][y] = np.log(result[1]+0.000001)
            matrix_share[x][y] = np.log(result[3]+0.000001)
            sample_share[x][y] = np.log(result[5]+0.000001)
            delete_share[x][y] = np.log(result[7]+0.000001)
            short_share[x][y] = np.log(result[9]+0.000001)
            long_share[x][y] = np.log(result[11]+0.000001)


fig, axs = plt.subplots(3, 2)

df_original = pd.DataFrame(original_share, range(len(lengths)), range(len(lengths)))
df_matrix = pd.DataFrame(matrix_share, range(len(lengths)), range(len(lengths)))
df_sample = pd.DataFrame(sample_share, range(len(lengths)), range(len(lengths)))
df_delete = pd.DataFrame(delete_share, range(len(lengths)), range(len(lengths)))
df_short = pd.DataFrame(short_share, range(len(lengths)), range(len(lengths)))
df_long = pd.DataFrame(long_share, range(len(lengths)), range(len(lengths)))

max_val = np.max([
    np.nanmax(np.ma.masked_equal(original_share, 0.0, copy=False)),
    np.nanmax(np.ma.masked_equal(matrix_share, 0.0, copy=False)),
    np.nanmax(np.ma.masked_equal(sample_share, 0.0, copy=False)),
    np.nanmax(np.ma.masked_equal(delete_share, 0.0, copy=False)),
    np.nanmax(np.ma.masked_equal(short_share, 0.0, copy=False)),
    np.nanmax(np.ma.masked_equal(long_share, 0.0, copy=False))
                  ])
min_val = np.min([
    np.nanmin(np.ma.masked_equal(original_share, 0.0, copy=False)),
    np.nanmin(np.ma.masked_equal(matrix_share, 0.0, copy=False)),
    np.nanmin(np.ma.masked_equal(sample_share, 0.0, copy=False)),
    np.nanmin(np.ma.masked_equal(delete_share, 0.0, copy=False)),
    np.nanmin(np.ma.masked_equal(short_share, 0.0, copy=False)),
    np.nanmin(np.ma.masked_equal(long_share, 0.0, copy=False))
])

axs[0][0] = sn.heatmap(df_original, annot=True, annot_kws={"size": 8},
                       ax=axs[0][0],
                       vmax=max_val,
                       vmin=min_val,
                       # vmax=np.nanmax(np.ma.masked_equal(original_share, 0.0, copy=False)),
                       # vmin=np.nanmin(np.ma.masked_equal(original_share, 0.0, copy=False)),
                       xticklabels=lengths, yticklabels=lengths) # font size
axs[0][0].set_title('original')

axs[0][1] = sn.heatmap(df_matrix, annot=True, annot_kws={"size": 8},
                       ax=axs[0][1],
                       vmax=max_val,
                       vmin=min_val,
                       # vmax=np.nanmax(np.ma.masked_equal(matrix_share, 0.0, copy=False)),
                       # vmin=np.nanmin(np.ma.masked_equal(matrix_share, 0.0, copy=False)),
                       xticklabels=lengths, yticklabels=lengths) # font size
axs[0][1].set_title('matrix')

axs[1][0] = sn.heatmap(df_sample, annot=True, annot_kws={"size": 8},
                       ax=axs[1][0],
                       vmax=max_val,
                       vmin=min_val,
                       # vmax=np.nanmax(np.ma.masked_equal(sample_share, 0.0, copy=False)),
                       # vmin=np.nanmin(np.ma.masked_equal(sample_share, 0.0, copy=False)),
                       xticklabels=lengths, yticklabels=lengths) # font size
axs[1][0].set_title('sample')

axs[1][1] = sn.heatmap(df_delete, annot=True, annot_kws={"size": 8},
                       ax=axs[1][1],
                       vmax=max_val,
                       vmin=min_val,
                       # vmax=np.nanmax(np.ma.masked_equal(delete_share, 0.0, copy=False)),
                       # vmin=np.nanmin(np.ma.masked_equal(delete_share, 0.0, copy=False)),
                       xticklabels=lengths, yticklabels=lengths) # font size
axs[1][1].set_title('matrix short')

axs[2][0] = sn.heatmap(df_short, annot=True, annot_kws={"size": 8},
                       ax=axs[2][0],
                       vmax=max_val,
                       vmin=min_val,
                       # vmax=np.nanmax(np.ma.masked_equal(short_share, 0.0, copy=False)),
                       # vmin=np.nanmin(np.ma.masked_equal(short_share, 0.0, copy=False)),
                       xticklabels=lengths, yticklabels=lengths) # font size
axs[2][0].set_title('short')

axs[2][1] = sn.heatmap(df_long, annot=True, annot_kws={"size": 8},
                       ax=axs[2][1],
                       vmax=max_val,
                       vmin=min_val,
                       # vmax=np.nanmax(np.ma.masked_equal(long_share, 0.0, copy=False)),
                       # vmin=np.nanmin(np.ma.masked_equal(long_share, 0.0, copy=False)),
                       xticklabels=lengths, yticklabels=lengths) # font size
axs[2][1].set_title('tf')
plt.suptitle("neurons-{}    repeats-{}".format(neurons, repeats))
plt.subplots_adjust(left=0, bottom=0,
                    right=1, top=0.95,
                    wspace=0.0, hspace=0.225)
plt.show()

print("done")
