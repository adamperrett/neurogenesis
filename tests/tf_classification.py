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

# load dataset
test = 'mpg'
if test == 'breast':
    from breast_data import *
    num_outputs = 2
    train_labels = training_set_labels
    train_feat = training_set_breasts
    num_inputs = len(train_feat[0])
    test_labels = test_set_labels
    test_feat = test_set_breasts
    retest_rate = 1
    retest_size = len(test_set_labels)
    X = np.array(train_feat + test_feat)
    Y = np.array(train_labels + test_labels)
    # X = pandas.DataFrame(train_feat + test_feat)
    # Y = pandas.DataFrame(train_labels + test_labels)
elif test == 'wine':
    from wine_data import *
    num_inputs = 13
    num_outputs = 3
    train_labels = training_set_labels
    train_feat = training_set_wines
    test_labels = test_set_labels
    test_feat = test_set_wines
    X = np.array(train_feat + test_feat)
    Y = np.array(train_labels + test_labels)
    # X = pandas.DataFrame(train_feat + test_feat)
    # Y = pandas.DataFrame(train_labels + test_labels)
elif test == 'mpg':
    from datasets.mpg_regression import norm_features, norm_mpg, min_mpg, max_mpg
    num_inputs = len(norm_features[0])
    num_outputs = 1
    retest_rate = 1
    retest_size = int(0.1 * len(norm_mpg))
    X = np.array(norm_features)
    Y = np.array(norm_mpg)
else:
    dataframe = pandas.read_csv("../datasets/iris.data", header=None)
    dataset = dataframe.values
    X = dataset[:,0:4].astype(float)
    Y = dataset[:,4]
    num_inputs = 4
    num_outputs = 3
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
if test == 'mpg':
    dummy_y = Y

# define baseline model
def baseline_model(n_neurons, lr):
    # create model
    model = Sequential()
    # model.add(GaussianNoise(stddev=k_stdev))
    model.add(Dense(n_neurons, input_dim=num_inputs, activation='relu'))
    # model.add(Dense(n_neurons, input_dim=num_inputs, activation='relu'))
    if test == 'mpg':
        # model.add(Dense(num_outputs, activation='sigmoid'))
        model.add(Dense(num_outputs, activation='linear'))
    else:
        model.add(Dense(num_outputs, activation='softmax'))
    # Compile model
    optimizer = tf.keras.optimizers.Adam(lr=lr)
    if test == 'mpg':
        loss = 'mean_squared_error'
    else:
        loss = 'categorical_crossentropy'
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model

num_neurons = 350
learning_rate = 0.008
batch_size = 64
epochs = 20
# noise_tests = np.linspace(0, 2., 21)
# k_stdev = K.variable(value=0.0)

splits = 10
testing_data = [[] for i in range(splits)]
training_data = [{} for i in range(splits)]
# all_noise_results = []

test_label = 'bp {} n{} lr{} b{}'.format(test, num_neurons, learning_rate, batch_size)

if test == 'mpg':
    sss = ShuffleSplit(n_splits=splits, test_size=0.1, random_state=2727)
else:
    sss = StratifiedKFold(n_splits=splits, random_state=2727, shuffle=True)
for repeat, (train_index, test_index) in enumerate(sss.split(X, Y)):
    net = baseline_model(num_neurons, learning_rate)
    if test == 'mpg':
        retest_callback = LambdaCallback(
            on_batch_end=lambda batch, logs:
            testing_data[repeat].append(
                np.average([np.square(a - b) for a, b in zip(net.predict(X[test_index]), Y[test_index])])))
    else:
        retest_callback = LambdaCallback(
            on_batch_end=lambda batch, logs:
            testing_data[repeat].append(
                np.average([np.argmax(a) == b for a, b in zip(net.predict(X[test_index]), Y[test_index])])))

    print("training model for repeat", repeat, test_label)
    history = net.fit(X[train_index], dummy_y[train_index],
                      batch_size=batch_size, validation_data=(X[test_index], dummy_y[test_index]),
                      callbacks=retest_callback,
                      epochs=epochs, verbose=True)

    print(test_label)
    for k in history.history.keys():
        training_data[repeat][k] = history.history[k]
        if 'acc' in k:
            print(k, history.history[k])
    # scce2 = tf.keras.losses.sparse_categorical_crossentropy(dummy_y[test_index], net.predict(X))

data_dict = {}

ave_test = []
for j in range(len(testing_data[0])):
    total = 0.
    for i in range(len(testing_data)):
        total += testing_data[i][j]
    ave_test.append(total / len(testing_data))
print(ave_test[:20])
print(ave_test[-20:])

data_dict['testing_data'] = testing_data
data_dict['ave_test'] = ave_test
data_dict['training_data'] = training_data

print('Averaged:')
ave_train = {}
for k in training_data[0]:
    total = np.zeros(epochs)
    for i in range(epochs):
        for r in range(len(training_data)):
            total[i] += training_data[r][k][i]
    ave_train[k] = total / len(training_data)
    print(k, ave_train[k])

print(test_label)
np.save("./data/{}".format(test_label), data_dict)


print('Done')