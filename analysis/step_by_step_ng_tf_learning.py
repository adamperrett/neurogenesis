import numpy as np
from scipy.special import softmax as sm
from copy import deepcopy
from models.neurogenesis import Network
from datasets.simple_tests import *
from models.convert_network import *
import random
import matplotlib
saving_plots = False
if saving_plots:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
# sns.set_theme(style="whitegrid")
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, LeaveOneOut, StratifiedKFold
# multi-class classification with Keras
import pandas
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GaussianNoise
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras import utils as np_utils
from tensorflow.keras import backend as K
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit, LeaveOneOut, StratifiedKFold, ShuffleSplit
from tensorflow.python.keras.callbacks import LambdaCallback
import numpy as np


# centres = [[.5, -.5],
#            [.5, .5],
#            [-.5, .5],
#            [-.5, -.5]]
centres = [[1, 0],
           [0, 0],
           # [0, 1]]#,
           [-1, 0]]
x_range = [-2, 2]
y_range = [-2, 2]
resolution = 100
spread = 0.2
examples = 100
test_set_size = 0.1
# simple_data, simple_labels = create_centroid_classes(centres, spread, examples)
# num_outputs = 2
num_inputs = 2
num_outputs = len(centres)
simple_data, simple_labels = create_bimodal_distribution(centres, spread, examples, max_classes=num_outputs)
train_labels = simple_labels[:int(examples*len(centres)*(1. - test_set_size))]
train_feat = simple_data[:int(examples*len(centres)*(1. - test_set_size))]
test_labels = simple_labels[int(examples*len(centres)*(1. - test_set_size)):]
test_feat = simple_data[int(examples*len(centres)*(1. - test_set_size)):]
retest_rate = 1
retest_size = len(test_feat)

sensitivity_width = 0.5
activation_threshold = 0.0
error_threshold = 0.0
maximum_synapses_per_neuron = 1
# fixed_hidden_amount = 0
fixed_hidden_ratio = 0.0
# fixed_hidden_ratio = fixed_hidden_amount / maximum_synapses_per_neuron
maximum_total_synapses = 100*3000000
input_spread = 0
activity_decay_rate = 1.#0.9999
activity_init = 1.
number_of_seeds = 0

maximum_net_size = int(maximum_total_synapses / maximum_synapses_per_neuron)
old_weight_modifier = 1.01
maturity = 100.
hidden_threshold = 0.95
delete_neuron_type = 'RL'
reward_decay = 0.9999
conv_size = 9
max_out_synapses = 50000
# activity_init = 1.0
always_inputs = False
replaying = False
error_type = 'sm'
repeats = 10
width_noise = 0.#5
noise_level = 0.#5
out_weight_scale = 0.0#0075
visualise_rate = 1
np.random.seed(27)
confusion_decay = 0.8
always_save = True
remove_class = 2
check_repeat = False
expecting = False
expect_type = 'oa'
surprise_threshold = 0.1

CLASSnet = Network(num_outputs, num_inputs,
                   error_threshold=error_threshold,
                   f_width=sensitivity_width,
                   width_noise=width_noise,
                   activation_threshold=activation_threshold,
                   maximum_total_synapses=maximum_total_synapses,
                   max_hidden_synapses=maximum_synapses_per_neuron,
                   activity_decay_rate=activity_decay_rate,
                   always_inputs=always_inputs,
                   old_weight_modifier=old_weight_modifier,
                   reward_decay=reward_decay,
                   delete_neuron_type=delete_neuron_type,
                   fixed_hidden_ratio=fixed_hidden_ratio,
                   activity_init=activity_init,
                   replaying=replaying,
                   hidden_threshold=hidden_threshold,
                   conv_size=conv_size,
                   surprise_threshold=surprise_threshold,
                   expecting=expecting,
                   check_repeat=check_repeat
                   )

def baseline_model(n_neurons, lr):
    # create model
    model = Sequential()
    # model.add(GaussianNoise(stddev=k_stdev))
    model.add(Dense(n_neurons, input_dim=num_inputs, activation='relu'))
    # model.add(Dense(n_neurons, input_dim=num_inputs, activation='relu'))
    model.add(Dense(num_outputs, activation='softmax'))
    # Compile model
    optimizer = tf.keras.optimizers.Adam(lr=lr)
    loss = 'categorical_crossentropy'
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model

num_neurons = 20
learning_rate = 0.01
batch_size = 1
epochs = 1

# encode class values as integers
X = train_feat
Y = train_labels
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
net = baseline_model(num_neurons, learning_rate)
testing_data = []

# retest_callback = LambdaCallback(
#     on_batch_end=lambda batch, logs:
#     testing_data.append(
#         np.average([np.argmax(a) == b for a, b in zip(net.predict(X), Y)])))

def train_tf_single(x, y, network):
    dumm_y = np.zeros(num_outputs)
    dumm_y[y] = 1
    dumm_y = np.vstack([dumm_y, dumm_y])
    x = np.vstack([x, x])
    h = network.fit(x, dumm_y,
                batch_size=batch_size,  # validation_data=(X, dummy_y),
                # callbacks=retest_callback,
                epochs=epochs, verbose=True
                )
    return h

def test_tf_full_grid(network):
    print("creating tf grid")
    class_choice = [{'x':[], 'y':[]} for i in range(num_outputs)]
    testing_data = []
    for x in np.linspace(x_range[0], x_range[1], resolution):
        for y in np.linspace(y_range[0], y_range[1], resolution):
            testing_data.append([x, y])
    print("predicting")
    output = network.predict(np.vstack(testing_data))
    print("choosing")
    for out, (x, y) in zip(output, testing_data):
        choice = int(np.argmax(out))
        class_choice[choice]['x'].append(x)
        class_choice[choice]['y'].append(y)
    return class_choice

def calculate_error(correct_class, activations):
    output_activations = np.zeros(num_outputs)
    error = np.zeros(num_outputs)
    one_hot_encoding = np.ones(num_outputs)
    one_hot_encoding *= -0
    one_hot_encoding[correct_class] = 1
    for output in range(num_outputs):
        output_activations[output] = activations['out{}'.format(output)]
    softmax = sm(output_activations)
    if min(softmax) != max(softmax):
        choice = softmax.argmax()
    else:
        choice = num_outputs
    for output in range(num_outputs):
        error[output] += softmax[output] - one_hot_encoding[output]
    return error, choice, softmax

def train_ng_single(x, y, network):
    activations = network.convert_inputs_to_activations(np.array(x))
    activations = network.response(activations)
    err, choice, s = calculate_error(y, activations)
    neuron_label = network.error_driven_neuro_genesis(
        activations, err,
        label=label)

def test_ng_single(x, y, network):
    activations = network.convert_inputs_to_activations(np.array(x))
    activations = network.response(activations)
    err, choice, s = calculate_error(y, activations)
    return choice

def test_ng_full_grid(network):
    print("creating ng grid")
    class_choice = [{'x':[], 'y':[]} for i in range(num_outputs)]
    for x in np.linspace(x_range[0], x_range[1], resolution):
        print("x:", x, "/", x_range[1])
        for y in np.linspace(y_range[0], y_range[1], resolution):
            choice = test_ng_single([x, y], 0, network)
            if choice != num_outputs:
                class_choice[choice]['x'].append(x)
                class_choice[choice]['y'].append(y)
    return class_choice


red = np.array([1, 0, 0])
green = np.array([0, 1, 0])
blue = np.array([0, 0, 1])
# colours = pl.cm.brg(np.linspace(0, 1, num_outputs))
colours = [red, green, blue]
dimness = 0.5
dim_colours = [red*dimness, green*dimness, blue*dimness]

tf_grid = test_tf_full_grid(net)
ng_grid = test_ng_full_grid(CLASSnet)
fig, ax = plt.subplots(1, 2)
ax[0].set_xlim(x_range)
ax[0].set_ylim(y_range)
ax[0].set_title("ReLU ANN")
ax[0].axis('off')
ax[1].set_xlim(x_range)
ax[1].set_ylim(y_range)
ax[1].set_title("SEED")
ax[1].axis('off')

for class_data, c in zip(tf_grid, dim_colours):
    ax[0].scatter(class_data['x'], class_data['y'], c=c)
for class_data, c in zip(ng_grid, dim_colours):
    ax[1].scatter(class_data['x'], class_data['y'], c=c)
split_x_by_y = [{'x':[], 'y':[]} for i in range(num_outputs)]
for x, y in zip(X, Y):
    split_x_by_y[y]['x'].append(x[0])
    split_x_by_y[y]['y'].append(x[1])
for class_data, c in zip(split_x_by_y, colours):
    ax[0].scatter(class_data['x'], class_data['y'], c=c)
    ax[1].scatter(class_data['x'], class_data['y'], c=c)
plt.show()

all_boundaries = []
all_boundaries.append([tf_grid, ng_grid])
show_points = [0, 3, 6, 10, 25, 100, 250]
for idx, (feature, label) in enumerate(zip(X, Y)):

    train_tf_single(feature, label, net)
    train_ng_single(feature, label, CLASSnet)

    print("finished test", idx)

    if idx+1 in show_points:
        print("Creating boundary for", idx-1)
        tf_grid = test_tf_full_grid(net)
        ng_grid = test_ng_full_grid(CLASSnet)
        all_boundaries.append([tf_grid, ng_grid])

        # fig, ax = plt.subplots(1, 2)
        # ax[0].set_xlim(x_range)
        # ax[0].set_ylim(y_range)
        # ax[0].set_title("ReLU ANN")
        # ax[0].axis('off')
        # ax[1].set_xlim(x_range)
        # ax[1].set_ylim(y_range)
        # ax[1].set_title("SEED")
        # ax[1].axis('off')
        # for class_data, c in zip(tf_grid, dim_colours):
        #     ax[0].scatter(class_data['x'], class_data['y'], c=c)
        # for class_data, c in zip(ng_grid, dim_colours):
        #     ax[1].scatter(class_data['x'], class_data['y'], c=c)
        # split_x_by_y = [{'x': [], 'y': []} for i in range(num_outputs)]
        # for x, y in zip(X, Y):
        #     split_x_by_y[y]['x'].append(x[0])
        #     split_x_by_y[y]['y'].append(x[1])
        # for class_data, c in zip(split_x_by_y, colours):
        #     ax[0].scatter(class_data['x'], class_data['y'], c=c)
        #     ax[1].scatter(class_data['x'], class_data['y'], c=c)
        # plt.show()
        print("catch")
        if idx+1 == show_points[-1]:
            fig, ax = plt.subplots(2, len(all_boundaries))
            for i, (tfg, ngg) in enumerate(all_boundaries):
                ax[0][i].set_xlim(x_range)
                ax[0][i].set_ylim(y_range)
                ax[0][i].set_title("Iteration:{}".format(show_points[i]))
                ax[0][i].axis('off')
                ax[1][i].set_xlim(x_range)
                ax[1][i].set_ylim(y_range)
                ax[1][i].axis('off')
                for class_data, c in zip(tfg, dim_colours):
                    ax[0][i].scatter(class_data['x'], class_data['y'], c=c)
                for class_data, c in zip(ngg, dim_colours):
                    ax[1][i].scatter(class_data['x'], class_data['y'], c=c)
                split_x_by_y = [{'x': [], 'y': []} for i in range(num_outputs)]
                for x, y in zip(X, Y):
                    split_x_by_y[y]['x'].append(x[0])
                    split_x_by_y[y]['y'].append(x[1])
                for class_data, c in zip(split_x_by_y, colours):
                    ax[0][i].scatter(class_data['x'], class_data['y'], c=c)
                    ax[1][i].scatter(class_data['x'], class_data['y'], c=c)
            matplotlib.pyplot.subplots_adjust(left=0, bottom=0,
                                              right=1, top=0.95,
                                              wspace=0.015, hspace=0)
            plt.show()
            print("done")

