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
sns.set_theme(style="whitegrid")
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
spread = 0.3
examples = 200
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

sensitivity_width = 0.4
activation_threshold = 0.0
error_threshold = 0.2
maximum_synapses_per_neuron = 13
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

num_neurons = 6
learning_rate = 0.0001
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

retest_callback = LambdaCallback(
    on_batch_end=lambda batch, logs:
    testing_data.append(
        np.average([np.argmax(a) == b for a, b in zip(net.predict(X), Y)])))

def train_tf_single(x, y, network):
    h = net.fit(x, y,
                batch_size=batch_size,  # validation_data=(X, dummy_y),
                callbacks=retest_callback,
                epochs=epochs, verbose=True
                )
    return h

def train_ng_single(x, y, net):
    activations = net.convert_inputs_to_activations(np.array(x))
    activations = net.response(activations)


for feature, label in zip(X, Y):

    history = network.fit(feature, label,
                          batch_size=batch_size, validation_data=(X, dummy_y),
                          callbacks=retest_callback,
                          epochs=epochs, verbose=True
                          )

    training_accuracy, training_classifications, \
            training_confusion, synapse_counts, \
            neuron_counts = test_net(CLASSnet, feature, label,
                                     indexes=train_index,
                                     test_net_label='Training',
                                     # fold_test_accuracy=fold_testing_accuracy,
                                     classifications=training_classifications,
                                     fold_string=fold_string,
                                     max_fold=maximum_fold_accuracy, noise_stdev=noise_level)
